import sqlite3
from typing import Dict, List, Tuple, Optional, Any
import json

class SQLiteTool:
    """Tool for interacting with SQLite database"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._schema_cache = None
    
    def get_schema(self, include_sample_data: bool = False) -> str:
        """Get database schema using PRAGMA
        
        Args:
            include_sample_data: If True, include sample rows from each table
        
        Returns:
            String representation of the database schema
        """
        if self._schema_cache:
            return self._schema_cache
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all tables (excluding internal SQLite tables)
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name;
        """)
        tables = cursor.fetchall()
        
        schema_parts = []
        for (table_name,) in tables:
            # Get column info - properly quote table name
            cursor.execute(f'PRAGMA table_info("{table_name}");')
            columns = cursor.fetchall()
            
            # Format: ColumnName Type
            col_defs = []
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                pk = " PRIMARY KEY" if col[5] else ""
                col_defs.append(f"  {col_name} {col_type}{pk}")
            
            table_def = f'"{table_name}"(\n' + ",\n".join(col_defs) + "\n)"
            schema_parts.append(table_def)
            
            # Optional: add sample data
            if include_sample_data:
                try:
                    cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 2;')
                    sample_rows = cursor.fetchall()
                    if sample_rows:
                        col_names = [col[1] for col in columns]
                        schema_parts.append(f"  Sample: {col_names[0]}={sample_rows[0][0]}")
                except:
                    pass
        
        conn.close()
        
        self._schema_cache = "\n\n".join(schema_parts)
        return self._schema_cache
    
    def get_table_names(self) -> List[str]:
        """Get list of all table names"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name;
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return tables
    
    def execute_query(self, sql: str, params: Optional[Tuple] = None) -> Dict[str, Any]:
        """Execute SQL query and return structured results
        
        Args:
            sql: SQL query string
            params: Optional parameters for parameterized queries
        
        Returns:
            Dict with keys: success, columns, rows, error, row_count
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            conn.close()
            
            return {
                "success": True,
                "columns": columns,
                "rows": rows,
                "error": None,
                "row_count": len(rows)
            }
            
        except Exception as e:
            return {
                "success": False,
                "columns": [],
                "rows": [],
                "error": str(e),
                "row_count": 0
            }
    
    def execute_query_json(self, sql: str) -> str:
        """Execute query and return results as JSON string"""
        result = self.execute_query(sql)
        
        if result["success"]:
            # Convert rows to list of dicts
            data = []
            for row in result["rows"]:
                row_dict = {}
                for i, col_name in enumerate(result["columns"]):
                    row_dict[col_name] = row[i]
                data.append(row_dict)
            
            return json.dumps({
                "success": True,
                "data": data,
                "row_count": result["row_count"]
            })
        else:
            return json.dumps({
                "success": False,
                "error": result["error"]
            })
    
    def validate_sql(self, sql: str) -> Tuple[bool, str]:
        """Validate SQL without executing (basic checks)
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        sql_upper = sql.upper().strip()
        
        # Only allow SELECT statements
        if not sql_upper.startswith('SELECT'):
            return False, "Only SELECT queries are allowed"
        
        # Disallow dangerous keywords
        dangerous = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE']
        for keyword in dangerous:
            if keyword in sql_upper:
                return False, f"Keyword {keyword} not allowed"
        
        # Check for balanced parentheses
        if sql.count('(') != sql.count(')'):
            return False, "Unbalanced parentheses"
        
        return True, ""
    
    def test_connection(self) -> bool:
        """Test if database connection works"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            result = cursor.fetchone()
            conn.close()
            return result[0] == 1
        except:
            return False
