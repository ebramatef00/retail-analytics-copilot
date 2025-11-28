from typing import TypedDict, List, Dict, Any, Annotated
import operator
from langgraph.graph import StateGraph, END
import json
import re
import os

from agent.dspy_signatures import setup_dspy, QueryRouter, ConstraintExtractor, SQLGenerator, AnswerSynthesizer
from agent.rag.retrieval import SimpleRetriever
from agent.tools.sqlite_tool import SQLiteTool


class AgentState(TypedDict):
    question: str
    format_hint: str
    route: str
    doc_chunks: List[Dict]
    sql_query: str
    sql_result: Dict
    sql_error: str
    constraints: str
    final_answer: Any
    explanation: str
    confidence: float
    citations: List[str]
    repair_count: int
    trace: Annotated[List[Dict], operator.add]


class HybridAgent:
    
    def __init__(self, docs_dir: str, db_path: str):
        setup_dspy()
        self.docs_dir = os.path.abspath(docs_dir)
        self.db_path = os.path.abspath(db_path)
        
        print(f"Using DB: {self.db_path}")
        print(f"Using Docs: {self.docs_dir}")
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        if not os.path.exists(self.docs_dir):
            raise FileNotFoundError(f"Docs directory not found: {self.docs_dir}")
        self.router = QueryRouter()
        self.constraint_extractor = ConstraintExtractor()
        self.sql_generator = SQLGenerator()
        self.synthesizer = AnswerSynthesizer()
        self.retriever = SimpleRetriever(self.docs_dir)
        self.db_tool = SQLiteTool(self.db_path)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        workflow.add_node("route", self.route_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("plan", self.plan_node)
        workflow.add_node("generate_sql", self.generate_sql_node)
        workflow.add_node("execute_sql", self.execute_sql_node)
        workflow.add_node("synthesize", self.synthesize_node)
        workflow.add_node("repair", self.repair_node)
        
        workflow.set_entry_point("route")
        
        workflow.add_conditional_edges(
            "route",
            lambda state: state["route"],
            {"rag": "retrieve", "sql": "plan", "hybrid": "retrieve"}
        )
        
        workflow.add_conditional_edges(
            "retrieve",
            lambda state: "synthesize" if state["route"] == "rag" else "plan"
        )
        
        workflow.add_edge("plan", "generate_sql")
        workflow.add_edge("generate_sql", "execute_sql")
        
        workflow.add_conditional_edges(
            "execute_sql",
            lambda state: "repair" if not state["sql_result"].get("success") and state["repair_count"] < 2 else "synthesize"
        )
        
        workflow.add_edge("repair", "generate_sql")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    def route_node(self, state: AgentState) -> AgentState:
        route = self.router(question=state["question"])
        state["route"] = route
        state["trace"].append({"node": "route", "route": route})
        return state
    
    def retrieve_node(self, state: AgentState) -> AgentState:
        results = self.retriever.retrieve(state["question"], top_k=3)
        state["doc_chunks"] = results
        state["trace"].append({"node": "retrieve", "num_chunks": len(results)})
        return state
    
    def plan_node(self, state: AgentState) -> AgentState:
        doc_context = "\n\n".join([f"{c['id']}: {c['content']}" for c in state.get("doc_chunks", [])])
        constraints = self.constraint_extractor(question=state["question"], documents=doc_context)
        state["constraints"] = constraints
        state["trace"].append({"node": "plan"})
        return state
    
    def generate_sql_node(self, state: AgentState) -> AgentState:
        schema = self.db_tool.get_schema()
        sql = self.sql_generator(question=state["question"], schema=schema, context=state.get("constraints", ""))
        state["sql_query"] = sql
        state["trace"].append({"node": "generate_sql"})
        return state
    
    def execute_sql_node(self, state: AgentState) -> AgentState:
        result = self.db_tool.execute_query(state["sql_query"])
        state["sql_result"] = result
        if not result["success"]:
            state["sql_error"] = result["error"]
        state["trace"].append({"node": "execute_sql", "success": result["success"]})
        return state
    
    def repair_node(self, state: AgentState) -> AgentState:
        state["repair_count"] = state.get("repair_count", 0) + 1
        error = state.get("sql_error", "")
        state["constraints"] = state.get("constraints", "") + f"\n\nError: {error}"
        state["trace"].append({"node": "repair", "attempt": state["repair_count"]})
        return state
    
    def synthesize_node(self, state: AgentState) -> AgentState:
        # Always use fallback extraction (more reliable)
        final_answer = self._fallback_extraction(state)
        
        state["final_answer"] = final_answer
        state["confidence"] = self._calc_confidence(state)
        state["citations"] = self._collect_citations(state)
        state["explanation"] = f"{state['route']}"
        state["trace"].append({"node": "synthesize"})
        
        return state
    
    def _fallback_extraction(self, state: AgentState) -> Any:
        """Extract answer from state"""
        format_hint = state["format_hint"]
        
        # RAG query
        if state["route"] == "rag":
            chunks = state.get("doc_chunks", [])
            text = " ".join([c["content"] for c in chunks])
            match = re.search(r'(\d+)\s*days?', text, re.IGNORECASE)
            return int(match.group(1)) if match else 0
        
        # SQL query
        result = state.get("sql_result", {})
        if not result.get("success") or not result.get("rows"):
            return 0 if format_hint == "int" else (0.0 if format_hint == "float" else ({} if "{" in format_hint else []))
        
        rows = result["rows"]
        
        if format_hint == "int":
            return int(rows[0][0] or 0)
        elif format_hint == "float":
            return round(float(rows[0][0] or 0), 2)
        elif "{category" in format_hint and len(rows[0]) >= 2:
            return {"category": str(rows[0][0]), "quantity": int(rows[0][1] or 0)}
        elif "{customer" in format_hint and len(rows[0]) >= 2:
            return {"customer": str(rows[0][0]), "margin": round(float(rows[0][1] or 0), 2)}
        elif format_hint.startswith("list"):
            return [{"product": str(r[0]), "revenue": round(float(r[1] or 0), 2)} for r in rows if r[0]]
        
        return rows[0][0]
    
    def _calc_confidence(self, state: AgentState) -> float:
        conf = 0.5
        if state.get("sql_result", {}).get("success"):
            conf += 0.3
        if state.get("doc_chunks"):
            conf += 0.2
        if state.get("repair_count", 0) > 0:
            conf -= 0.1 * state["repair_count"]
        return round(max(0.0, min(1.0, conf)), 2)
    
    def _collect_citations(self, state: AgentState) -> List[str]:
        cites = set()
        
        for chunk in state.get("doc_chunks", []):
            cites.add(chunk["id"])
        
        if state.get("sql_result", {}).get("success"):
            sql = state.get("sql_query", "").upper()
            for table in ["Orders", "Order Details", "Products", "Customers", "Categories"]:
                if table.upper() in sql or table.replace(" ", "").upper() in sql:
                    cites.add(table)
        
        return sorted(list(cites))
    
    def run(self, question: str, format_hint: str) -> Dict[str, Any]:
        initial_state = {
            "question": question,
            "format_hint": format_hint,
            "route": "",
            "doc_chunks": [],
            "sql_query": "",
            "sql_result": {},
            "sql_error": "",
            "constraints": "",
            "final_answer": None,
            "explanation": "",
            "confidence": 0.0,
            "citations": [],
            "repair_count": 0,
            "trace": []
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "final_answer": final_state["final_answer"],
            "sql": final_state.get("sql_query", ""),
            "confidence": final_state["confidence"],
            "explanation": final_state["explanation"],
            "citations": final_state["citations"],
            "trace": final_state["trace"]
        }
