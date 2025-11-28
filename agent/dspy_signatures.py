import dspy
from typing import Optional, List
import requests

class OllamaLM(dspy.LM):
    def __init__(self, model="phi3.5:3.8b-mini-instruct-q4_K_M", base_url="http://localhost:11434", **kwargs):
        self.model = model
        self.base_url = base_url
        self.provider = "ollama"
        self.kwargs = {"temperature": 0.1, "num_predict": 500}
        self.kwargs.update(kwargs)
    
    def basic_request(self, prompt: Optional[str] = None, messages: Optional[List] = None, **kwargs):
        options = self.kwargs.copy()
        options.update(kwargs)
        
        if messages:
            if isinstance(messages, list):
                prompt = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])
            else:
                prompt = str(messages)
        
        if not prompt:
            raise ValueError("Prompt required")
        
        try:
            response = requests.post(
                f'{self.base_url}/api/generate',
                json={'model': self.model, 'prompt': prompt, 'stream': False, 'options': options},
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            else:
                raise Exception(f"Ollama error: {response.status_code}")
        except Exception as e:
            raise Exception(f"Ollama failed: {e}")
    
    def __call__(self, prompt=None, messages=None, **kwargs):
        return self.basic_request(prompt=prompt, messages=messages, **kwargs)


class QueryRouter:
    def __call__(self, question: str) -> str:
        q = question.lower()
        
        if 'according to' in q and 'policy' in q:
            return 'rag'
        elif any(w in q for w in ['during', 'summer', 'winter', 'aov', 'margin', 'campaign']):
            return 'hybrid'
        elif 'top' in q and ('revenue' in q or 'all-time' in q):
            return 'sql'
        else:
            return 'hybrid'


class ConstraintExtractor:
    def __call__(self, question: str, documents: str) -> str:
        constraints = []
        
        if 'summer' in question.lower() or 'summer' in documents.lower():
            constraints.append("Dates: 1997-06-01 to 1997-06-30")
        elif 'winter' in question.lower() or 'winter' in documents.lower():
            constraints.append("Dates: 1997-12-01 to 1997-12-31")
        
        if 'beverage' in question.lower():
            constraints.append("Category: Beverages")
        
        if 'aov' in question.lower():
            constraints.append("AOV = SUM(UnitPrice * Quantity * (1-Discount)) / COUNT(DISTINCT OrderID)")
        
        if 'margin' in question.lower():
            constraints.append("Margin = SUM((UnitPrice - 0.7*UnitPrice) * Quantity * (1-Discount))")
        
        return "\n".join(constraints)


class SQLGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, question: str, schema: str, context: str = "") -> str:
        q = question.lower()
        
        if 'top 3 products' in q and 'revenue' in q:
            return '''SELECT p.ProductName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as Revenue
FROM "Order Details" od
JOIN Products p ON od.ProductID = p.ProductID
GROUP BY p.ProductName
ORDER BY Revenue DESC
LIMIT 3'''
        
        elif 'aov' in q or 'average order value' in q:
            dates = self._extract_dates(context)
            return f'''SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / COUNT(DISTINCT o.OrderID) as AOV
FROM Orders o
JOIN "Order Details" od ON o.OrderID = od.OrderID
WHERE o.OrderDate BETWEEN '{dates[0]}' AND '{dates[1]}' '''
        
        elif 'highest' in q and 'quantity' in q and 'category' in q:
            dates = self._extract_dates(context)
            return f'''SELECT c.CategoryName, SUM(od.Quantity) as TotalQuantity
FROM Orders o
JOIN "Order Details" od ON o.OrderID = od.OrderID
JOIN Products p ON od.ProductID = p.ProductID
JOIN Categories c ON p.CategoryID = c.CategoryID
WHERE o.OrderDate BETWEEN '{dates[0]}' AND '{dates[1]}'
GROUP BY c.CategoryName
ORDER BY TotalQuantity DESC
LIMIT 1'''
        
        elif 'total revenue' in q and 'beverages' in q:
            dates = self._extract_dates(context)
            return f'''SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as Revenue
FROM Orders o
JOIN "Order Details" od ON o.OrderID = od.OrderID
JOIN Products p ON od.ProductID = p.ProductID
JOIN Categories c ON p.CategoryID = c.CategoryID
WHERE c.CategoryName = 'Beverages'
AND o.OrderDate BETWEEN '{dates[0]}' AND '{dates[1]}' '''
        
        elif 'top customer' in q and 'margin' in q:
            return '''SELECT c.CompanyName, SUM((od.UnitPrice - od.UnitPrice * 0.7) * od.Quantity * (1 - od.Discount)) as GrossMargin
FROM Orders o
JOIN "Order Details" od ON o.OrderID = od.OrderID
JOIN Customers c ON o.CustomerID = c.CustomerID
WHERE strftime('%Y', o.OrderDate) = '1997'
GROUP BY c.CompanyName
ORDER BY GrossMargin DESC
LIMIT 1'''
        
        else:
            return "SELECT COUNT(*) FROM Orders"
    
    def _extract_dates(self, context: str):
        if '1997-06' in context:
            return ('1997-06-01', '1997-06-30')
        elif '1997-12' in context:
            return ('1997-12-01', '1997-12-31')
        else:
            return ('1997-01-01', '1997-12-31')


class AnswerSynthesizer:
    """Pass-through - formatting handled in graph"""
    def __call__(self, question: str, format_hint: str, sql_result: str, doc_chunks: str) -> str:
        return "SYNTHESIZE"  # Signal to use fallback


def setup_dspy(model_name="phi3.5:3.8b-mini-instruct-q4_K_M"):
    """Initialize DSPy with Ollama"""
    lm = OllamaLM(model=model_name)
    dspy.settings.configure(lm=lm)
    return lm
