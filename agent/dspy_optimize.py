import dspy
from dspy.teleprompt import BootstrapFewShot
from typing import Union, Any, Callable
TRAIN_SQL_DATA = [
    dspy.Example(
        db_schema="CREATE TABLE Orders (OrderID INT, CustomerID INT, OrderDate DATE, Freight FLOAT)",
        question="How many orders were placed in 1997?",
        sql_query="SELECT COUNT(*) FROM Orders WHERE YEAR(OrderDate) = 1997"
    ).with_inputs("db_schema", "question"),
    dspy.Example(
        db_schema="CREATE TABLE Products (ProductID INT, ProductName VARCHAR, Price FLOAT); CREATE TABLE Orders (OrderID INT, OrderDate DATE)",
        question="What is the price of the most expensive product?",
        sql_query="SELECT MAX(Price) FROM Products"
    ).with_inputs("db_schema", "question"),
    dspy.Example(
        db_schema="CREATE TABLE Customers (CustomerID INT, CustomerName VARCHAR, Country VARCHAR)",
        question="How many unique customers are there?",
        sql_query="SELECT COUNT(DISTINCT CustomerID) FROM Customers"
    ).with_inputs("db_schema", "question"),
]


TEST_SQL_DATA = [
    dspy.Example(
        db_schema="CREATE TABLE Orders (OrderID INT, CustomerID INT, OrderDate DATE)",
        question="Count the total number of orders placed in 1997.",
        sql_query="SELECT COUNT(*) FROM Orders WHERE YEAR(OrderDate) = 1997"
    ).with_inputs("db_schema", "question"),
    dspy.Example(
        db_schema="CREATE TABLE Products (ProductID INT, ProductName VARCHAR, Price FLOAT)",
        question="Show me the most expensive product name.",
        sql_query="SELECT ProductName FROM Products ORDER BY Price DESC LIMIT 1"
    ).with_inputs("db_schema", "question"),
]

def sql_metric(example: dspy.Example, pred: Any, trace: Any = None) -> float:
    generated_sql = pred.sql_query.strip().lower()
    expected_sql = example.sql_query.strip().lower()
    has_select = "select" in generated_sql
    has_valid_structure = has_select and ("from" in generated_sql or "count" in generated_sql)
    matches_expected = generated_sql == expected_sql
    if matches_expected:
        return 1.0
    elif has_valid_structure:
        return 0.5
    else:
        return 0.0


def optimize_sql_generator(base_sql_module: Union[dspy.Module, Callable]) -> Union[dspy.Module, Callable]:
    is_wrapper = hasattr(base_sql_module, "generate")
    target_module = base_sql_module.generate if is_wrapper else base_sql_module

    if not callable(target_module):
        raise TypeError(f"Provided SQL generator is not callable. Type: {type(target_module)}")

    eval_kwargs = dict(
        devset=TEST_SQL_DATA,
        metric=sql_metric,
        display_progress=False,
        display_table=0
    )

    try:
        dspy.evaluate.evaluate(target_module, **eval_kwargs).average_metric
    except Exception:
        return base_sql_module

    try:
        teleprompter = BootstrapFewShot(metric=sql_metric, max_bootstrapped_demos=2)
        optimized_module = teleprompter.compile(target_module, trainset=TRAIN_SQL_DATA)
    except Exception:
        return base_sql_module

    try:
        dspy.evaluate.evaluate(optimized_module, **eval_kwargs).average_metric
        if is_wrapper:
            base_sql_module.generate = optimized_module
            return base_sql_module
        else:
            return optimized_module
    except Exception:
        return base_sql_module