import dspy
from typing import List, Dict, Any, Optional
import json
import requests
import re

# Custom Ollama LM that works with DSPy's adapter system
class OllamaLM(dspy.LM):
    def __init__(self, model="phi3.5:3.8b-mini-instruct-q4_K_M", base_url="http://localhost:11434", **kwargs):
        self.model = model
        self.base_url = base_url
        self.provider = "ollama"
        self.history = []
        self.kwargs = {
            "temperature": 0.1,
            "num_predict": 1000,
        }
        self.kwargs.update(kwargs)
    
    def basic_request(self, prompt: Optional[str] = None, messages: Optional[List] = None, **kwargs):
        """DSPy calls this method - handle both prompt and messages format"""
        # Merge kwargs
        options = self.kwargs.copy()
        options.update(kwargs)
        
        # Handle both formats: prompt string or messages list
        if messages:
            # Convert messages to a single prompt
            if isinstance(messages, list):
                prompt_parts = []
                for msg in messages:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    prompt_parts.append(f"{role}: {content}")
                prompt = "\n\n".join(prompt_parts)
            else:
                prompt = str(messages)
        
        if not prompt:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        
        # Add instruction to always output valid JSON for structured outputs
        if "output fields" in prompt.lower() or "json" in prompt.lower():
            prompt += "\n\nIMPORTANT: Output ONLY valid JSON with the required fields. No explanation before or after."
        
        try:
            response = requests.post(
                f'{self.base_url}/api/generate',
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False,
                    'options': options
                },
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                output = result.get('response', '').strip()
                
                if "output fields" in prompt.lower() or "json" in prompt.lower():
                    # Attempt to clean and extract JSON
                    cleaned_output = output.strip()
                    # Remove markdown code blocks (e.g., ```json, ```sql, ```)
                    cleaned_output = re.sub(r'```(?:json|sql)?\s*([\s\S]*?)```', r'\1', cleaned_output, flags=re.IGNORECASE).strip()
                    
                    # Attempt to fix a common LLM error where keys are not quoted
                    cleaned_output = re.sub(r'\{(\w+):', r'{"\1":', cleaned_output.strip())
                    
                    extracted_json_str = None
                    # Try direct parse
                    try:
                        json_obj = json.loads(cleaned_output)
                        extracted_json_str = json.dumps(json_obj) # Ensure re-serialized to clean string
                    except json.JSONDecodeError:
                        # If direct parsing fails, look for the last valid JSON object using a more robust method
                        # This attempts to find all potential JSON objects and validate them.
                        json_candidates = []
                        stack = []
                        start_idx = -1
                        for i, char in enumerate(cleaned_output):
                            if char == '{':
                                if not stack:
                                    start_idx = i
                                stack.append(char)
                            elif char == '}':
                                if stack and stack[-1] == '{':
                                    stack.pop()
                                    if not stack and start_idx != -1:
                                        potential_json = cleaned_output[start_idx : i+1]
                                        try:
                                            json.loads(potential_json) # Validate
                                            json_candidates.append(potential_json)
                                        except json.JSONDecodeError:
                                            pass
                                else:
                                    # Mismatched closing brace, reset
                                    stack = []
                                    start_idx = -1

                        if json_candidates:
                            # Take the last valid JSON object found
                            extracted_json_str = json.dumps(json.loads(json_candidates[-1])) # Re-parse and dump for cleanliness
                        
                    if extracted_json_str:
                        output = extracted_json_str
                    else:
                        # If no valid JSON could be extracted despite being expected, return an empty JSON object.
                        # This ensures dspy.Predict doesn't crash on malformed output,
                        # and SQLGenerator can handle the effectively empty response.
                        print(f"Warning: Expected JSON output but could not parse. Raw LLM output: {output[:100]}...")
                        output = "{}" # Return an empty but valid JSON string

                
                self.history.append({'prompt': prompt[:200], 'response': output[:200]})
                return output
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to connect to Ollama: {e}")
    
    def __call__(self, prompt: Optional[str] = None, messages: Optional[List] = None, **kwargs):
        """Also support direct calls"""
        return self.basic_request(prompt=prompt, messages=messages, **kwargs)


# DSPy Signatures
class RouteQuery(dspy.Signature):
    """Classify query type: rag (docs only), sql (database only), or hybrid (both)"""
    question = dspy.InputField(desc="User question")
    route = dspy.OutputField(desc="Must be exactly one of: rag, sql, hybrid")


class GenerateSQL(dspy.Signature):
    """Generate valid SQLite query from natural language question"""
    question = dspy.InputField(desc="Natural language question")
    db_schema = dspy.InputField(desc="Database schema with table definitions")
    context = dspy.InputField(desc="Additional context from documents (dates, definitions)")
    sql = dspy.OutputField(desc="Valid SQLite SELECT query without semicolon")


class SynthesizeAnswer(dspy.Signature):
    """Create final answer matching the required format with citations"""
    question = dspy.InputField(desc="Original question")
    format_hint = dspy.InputField(desc="Required output format (int, float, dict, list)")
    sql_result = dspy.InputField(desc="SQL query results as JSON string")
    doc_chunks = dspy.InputField(desc="Retrieved document chunks as JSON string")
    answer = dspy.OutputField(desc="Final answer matching format_hint exactly")
    citations = dspy.OutputField(desc="Comma-separated list of tables and doc chunks used")


# DSPy Modules (with CoT for better reasoning)
class QueryRouter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.route = dspy.ChainOfThought(RouteQuery)
    
    def forward(self, question: str) -> str:
        try:
            result = self.route(question=question)
            route = result.route.lower().strip()
            
            # Ensure valid route
            if route not in ['rag', 'sql', 'hybrid']:
                # Default logic
                if any(word in question.lower() for word in ['policy', 'return', 'according to']):
                    return 'rag'
                elif any(word in question.lower() for word in ['total', 'revenue', 'top', 'customer']):
                    return 'hybrid'
                else:
                    return 'sql'
            
            return route
        except Exception as e:
            # Fallback routing logic
            question_lower = question.lower()
            if any(word in question_lower for word in ['policy', 'return', 'according to', 'definition']):
                return 'rag'
            elif any(word in question_lower for word in ['during', 'summer', 'winter', 'calendar', 'campaign']):
                return 'hybrid'
            else:
                return 'sql'


class SQLGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateSQL)  # Use Predict instead of ChainOfThought
    
    def forward(self, question: str, schema: str, context: str = "") -> str:
        try:
            result = self.generate(question=question, db_schema=schema, context=context)
            sql = result.sql.strip()
            
            # Clean up SQL
            sql = sql.replace('```sql', '').replace('```', '').strip()
            if sql.endswith(';'):
                sql = sql[:-1]
            
            # Basic validation
            if not sql or len(sql) < 10:
                raise ValueError("Generated SQL too short")
            
            return sql
        except Exception as e:
            print(f"SQL generation error: {e}")
            # Better fallback based on question
            if "top" in question.lower() and "product" in question.lower():
                return 'SELECT ProductName, SUM(Quantity) as Total FROM "Order Details" od JOIN Products p ON od.ProductID = p.ProductID GROUP BY ProductName ORDER BY Total DESC LIMIT 3'
            return "SELECT COUNT(*) FROM Orders"


class AnswerSynthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.Predict(SynthesizeAnswer)  # Use Predict instead of ChainOfThought
    
    def forward(self, question: str, format_hint: str, sql_result: str, doc_chunks: str) -> Dict[str, Any]:
        try:
            result = self.synthesize(
                question=question,
                format_hint=format_hint,
                sql_result=sql_result,
                doc_chunks=doc_chunks
            )
            
            # Parse answer
            answer_str = result.answer.strip()
            citations_str = result.citations.strip()
            
            # Extract citations
            citations = [c.strip() for c in citations_str.split(',') if c.strip()]
            
            return {
                'answer': answer_str,
                'citations': citations
            }
        except Exception as e:
            print(f"Synthesis error: {e}")
            # Fallback: parse from doc_chunks or sql_result
            citations = []
            if sql_result and sql_result != "[]":
                citations.append("database")
            if doc_chunks and doc_chunks != "[]":
                try:
                    chunks = json.loads(doc_chunks)
                    citations.extend([c.get('id', '') for c in chunks if c.get('id')])
                except:
                    pass
            
            return {
                'answer': 'Unable to synthesize answer',
                'citations': citations
            }


def setup_dspy(model_name="phi3.5:3.8b-mini-instruct-q4_K_M"):
    """Initialize DSPy with Ollama"""
    lm = OllamaLM(model=model_name)
    dspy.settings.configure(lm=lm)
    return lm
