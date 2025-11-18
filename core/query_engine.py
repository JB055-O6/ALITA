"""
Natural Language Database Query Engine
Advanced SQL generation and execution with AI-powered insights
Uses state-of-the-art free models and frameworks
"""

import os
import re
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available - SQL generation will be limited")

try:
    import sqlalchemy
    from sqlalchemy import create_engine, inspect, text
    from sqlalchemy.pool import NullPool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logging.warning("SQLAlchemy not available - advanced DB support limited")

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logging.warning("DuckDB not available - analytics features limited")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available - visualizations limited")


class DatabaseType(Enum):
    """Supported database types"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    DUCKDB = "duckdb"


class QueryMode(Enum):
    """Query execution modes"""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"


@dataclass
class QueryResult:
    """Result of a database query"""
    success: bool
    data: Optional[pd.DataFrame]
    sql_query: str
    execution_time: float
    row_count: int
    error: Optional[str] = None
    insights: Optional[List[str]] = None
    visualization: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DatabaseSchema:
    """Database schema information"""
    tables: List[str]
    columns: Dict[str, List[Dict[str, str]]]
    relationships: List[Dict[str, str]]
    indexes: Dict[str, List[str]]
    statistics: Dict[str, Any]


class QueryEngine:
    """
    Advanced Natural Language to SQL Query Engine
    
    Features:
    - Multi-database support (SQLite, PostgreSQL, MySQL, DuckDB)
    - AI-powered SQL generation using SQLCoder/Llama models
    - Safe query execution with read-only mode
    - Automatic data visualization
    - Intelligent insight generation
    - Query disambiguation
    - Schema understanding
    - Query optimization
    """
    
    def __init__(
        self,
        model_name: str = "defog/sqlcoder-7b-2",
        device: str = "auto",
        default_mode: QueryMode = QueryMode.READ_ONLY,
        load_model: bool = True
    ):
        """
        Initialize Query Engine
        
        Args:
            model_name: HuggingFace model for SQL generation
            device: Device for model inference
            default_mode: Default query execution mode
            load_model: Whether to load AI model (set False for testing)
        """
        self.logger = logging.getLogger(__name__)
        self.default_mode = default_mode
        self.connections: Dict[str, Any] = {}
        self.schemas: Dict[str, DatabaseSchema] = {}
        
        # Initialize SQL generation model
        self.model = None
        self.tokenizer = None
        self.device = self._setup_device(device)
        
        if TRANSFORMERS_AVAILABLE and load_model:
            self._load_sql_model(model_name)
        
        # Query history for learning
        self.query_history: List[Dict[str, Any]] = []
        
        # Insight generation patterns
        self.insight_patterns = self._initialize_insight_patterns()
        
        self.logger.info("QueryEngine initialized successfully")
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_sql_model(self, model_name: str):
        """Load SQL generation model"""
        try:
            self.logger.info(f"Loading SQL model: {model_name}")
            
            # Use smaller model if GPU memory is limited
            if self.device == "cuda":
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory < 6:
                    # Use smaller model for limited VRAM
                    model_name = "NumbersStation/nsql-llama-2-7B"
                    self.logger.info(f"Using smaller model for limited VRAM: {model_name}")
            
            # Try to load with 4-bit quantization if available
            try:
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
            except ImportError:
                # Fall back to regular loading without quantization
                self.logger.info("BitsAndBytes not available, loading model without quantization")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                
                if self.device != "auto":
                    self.model = self.model.to(self.device)
            
            self.logger.info("SQL model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load SQL model: {e}")
            self.logger.info("Falling back to rule-based SQL generation")
    
    def _initialize_insight_patterns(self) -> Dict[str, Any]:
        """Initialize patterns for insight generation"""
        return {
            "trends": {
                "increasing": lambda df, col: df[col].is_monotonic_increasing,
                "decreasing": lambda df, col: df[col].is_monotonic_decreasing,
                "seasonal": lambda df, col: self._detect_seasonality(df, col)
            },
            "anomalies": {
                "outliers": lambda df, col: self._detect_outliers(df, col),
                "missing": lambda df, col: df[col].isna().sum() / len(df) > 0.1
            },
            "correlations": {
                "strong_positive": lambda df, cols: df[cols].corr().abs() > 0.7,
                "strong_negative": lambda df, cols: df[cols].corr() < -0.7
            }
        }
    
    def connect_database(
        self,
        db_type: DatabaseType,
        connection_string: str,
        name: str = "default"
    ) -> bool:
        """
        Connect to a database
        
        Args:
            db_type: Type of database
            connection_string: Connection string or file path
            name: Connection name
            
        Returns:
            Success status
        """
        try:
            if db_type == DatabaseType.SQLITE:
                conn = sqlite3.connect(connection_string)
                self.connections[name] = {
                    "type": db_type,
                    "connection": conn,
                    "path": connection_string
                }
            
            elif db_type == DatabaseType.DUCKDB and DUCKDB_AVAILABLE:
                conn = duckdb.connect(connection_string)
                self.connections[name] = {
                    "type": db_type,
                    "connection": conn,
                    "path": connection_string
                }
            
            elif SQLALCHEMY_AVAILABLE:
                engine = create_engine(
                    connection_string,
                    poolclass=NullPool,
                    connect_args={"options": "-c default_transaction_read_only=on"}
                    if self.default_mode == QueryMode.READ_ONLY else {}
                )
                self.connections[name] = {
                    "type": db_type,
                    "connection": engine,
                    "path": connection_string
                }
            
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            # Load schema
            self.schemas[name] = self._load_schema(name)
            
            self.logger.info(f"Connected to {db_type.value} database: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    def _load_schema(self, connection_name: str) -> DatabaseSchema:
        """Load database schema information"""
        try:
            conn_info = self.connections[connection_name]
            conn = conn_info["connection"]
            db_type = conn_info["type"]
            
            if db_type == DatabaseType.SQLITE:
                return self._load_sqlite_schema(conn)
            elif db_type == DatabaseType.DUCKDB:
                return self._load_duckdb_schema(conn)
            elif SQLALCHEMY_AVAILABLE:
                return self._load_sqlalchemy_schema(conn)
            
        except Exception as e:
            self.logger.error(f"Failed to load schema: {e}")
            return DatabaseSchema([], {}, [], {}, {})
    
    def _load_sqlite_schema(self, conn) -> DatabaseSchema:
        """Load SQLite schema"""
        cursor = conn.cursor()
        
        # Get tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Get columns for each table
        columns = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns[table] = [
                {"name": row[1], "type": row[2], "nullable": not row[3]}
                for row in cursor.fetchall()
            ]
        
        # Get indexes
        indexes = {}
        for table in tables:
            cursor.execute(f"PRAGMA index_list({table})")
            indexes[table] = [row[1] for row in cursor.fetchall()]
        
        return DatabaseSchema(
            tables=tables,
            columns=columns,
            relationships=[],
            indexes=indexes,
            statistics={}
        )
    
    def _load_duckdb_schema(self, conn) -> DatabaseSchema:
        """Load DuckDB schema"""
        # Get tables
        tables_df = conn.execute("SHOW TABLES").fetchdf()
        tables = tables_df['name'].tolist() if not tables_df.empty else []
        
        # Get columns
        columns = {}
        for table in tables:
            cols_df = conn.execute(f"DESCRIBE {table}").fetchdf()
            columns[table] = [
                {"name": row['column_name'], "type": row['column_type'], "nullable": row['null'] == 'YES'}
                for _, row in cols_df.iterrows()
            ]
        
        return DatabaseSchema(
            tables=tables,
            columns=columns,
            relationships=[],
            indexes={},
            statistics={}
        )
    
    def _load_sqlalchemy_schema(self, engine) -> DatabaseSchema:
        """Load schema using SQLAlchemy"""
        inspector = inspect(engine)
        
        tables = inspector.get_table_names()
        
        columns = {}
        for table in tables:
            cols = inspector.get_columns(table)
            columns[table] = [
                {"name": col['name'], "type": str(col['type']), "nullable": col['nullable']}
                for col in cols
            ]
        
        # Get foreign keys
        relationships = []
        for table in tables:
            fks = inspector.get_foreign_keys(table)
            for fk in fks:
                relationships.append({
                    "from_table": table,
                    "to_table": fk['referred_table'],
                    "columns": fk['constrained_columns']
                })
        
        # Get indexes
        indexes = {}
        for table in tables:
            idx = inspector.get_indexes(table)
            indexes[table] = [i['name'] for i in idx]
        
        return DatabaseSchema(
            tables=tables,
            columns=columns,
            relationships=relationships,
            indexes=indexes,
            statistics={}
        )
    
    def natural_language_query(
        self,
        query: str,
        connection_name: str = "default",
        mode: Optional[QueryMode] = None
    ) -> QueryResult:
        """
        Execute natural language query
        
        Args:
            query: Natural language query
            connection_name: Database connection to use
            mode: Query execution mode
            
        Returns:
            Query result with data and insights
        """
        import time
        start_time = time.time()
        
        try:
            # Check if query needs disambiguation
            if self._needs_disambiguation(query):
                clarification = self._generate_clarification(query, connection_name)
                return QueryResult(
                    success=False,
                    data=None,
                    sql_query="",
                    execution_time=0,
                    row_count=0,
                    error=f"Query needs clarification: {clarification}"
                )
            
            # Generate SQL
            sql_query = self._generate_sql(query, connection_name)
            
            if not sql_query:
                return QueryResult(
                    success=False,
                    data=None,
                    sql_query="",
                    execution_time=0,
                    row_count=0,
                    error="Failed to generate SQL query"
                )
            
            # Validate query safety
            if not self._is_query_safe(sql_query, mode or self.default_mode):
                return QueryResult(
                    success=False,
                    data=None,
                    sql_query=sql_query,
                    execution_time=0,
                    row_count=0,
                    error="Query contains unsafe operations"
                )
            
            # Execute query
            result_df = self._execute_query(sql_query, connection_name)
            
            execution_time = time.time() - start_time
            
            # Generate insights
            insights = self._generate_insights(result_df, query)
            
            # Create visualization
            visualization = self._create_visualization(result_df, query)
            
            # Store in history
            self.query_history.append({
                "query": query,
                "sql": sql_query,
                "timestamp": time.time(),
                "success": True
            })
            
            return QueryResult(
                success=True,
                data=result_df,
                sql_query=sql_query,
                execution_time=execution_time,
                row_count=len(result_df),
                insights=insights,
                visualization=visualization,
                metadata={
                    "connection": connection_name,
                    "mode": (mode or self.default_mode).value
                }
            )
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            return QueryResult(
                success=False,
                data=None,
                sql_query=sql_query if 'sql_query' in locals() else "",
                execution_time=time.time() - start_time,
                row_count=0,
                error=str(e)
            )
    
    def _needs_disambiguation(self, query: str) -> bool:
        """Check if query needs clarification"""
        ambiguous_terms = [
            "that", "this", "those", "these",
            "recent", "latest", "best", "worst",
            "some", "few", "many"
        ]
        
        query_lower = query.lower()
        return any(term in query_lower for term in ambiguous_terms)
    
    def _generate_clarification(self, query: str, connection_name: str) -> str:
        """Generate clarification question"""
        schema = self.schemas.get(connection_name)
        
        if not schema:
            return "Please specify which database table you want to query."
        
        # Analyze query for ambiguities
        if "recent" in query.lower() or "latest" in query.lower():
            return "How recent? Please specify a time period (e.g., last 7 days, last month)."
        
        if "best" in query.lower() or "worst" in query.lower():
            return "Best/worst by what metric? Please specify the criteria."
        
        # Check if table is ambiguous
        query_words = set(query.lower().split())
        matching_tables = [t for t in schema.tables if t.lower() in query_words]
        
        if len(matching_tables) > 1:
            return f"Which table? Options: {', '.join(matching_tables)}"
        
        return "Please provide more specific details about your query."
    
    def _generate_sql(self, query: str, connection_name: str) -> str:
        """Generate SQL from natural language"""
        schema = self.schemas.get(connection_name)
        
        if not schema:
            self.logger.error("No schema available for SQL generation")
            return ""
        
        if self.model and self.tokenizer:
            return self._generate_sql_with_model(query, schema)
        else:
            return self._generate_sql_rule_based(query, schema)
    
    def _generate_sql_with_model(self, query: str, schema: DatabaseSchema) -> str:
        """Generate SQL using AI model"""
        try:
            # Build prompt with schema context
            schema_text = self._format_schema_for_prompt(schema)
            
            prompt = f"""### Task
Generate a SQL query to answer the following question.

### Database Schema
{schema_text}

### Question
{query}

### SQL Query
"""
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract SQL from generated text
            sql_query = self._extract_sql_from_text(generated_text)
            
            return sql_query
            
        except Exception as e:
            self.logger.error(f"Model-based SQL generation failed: {e}")
            return self._generate_sql_rule_based(query, schema)
    
    def _format_schema_for_prompt(self, schema: DatabaseSchema) -> str:
        """Format schema for model prompt"""
        schema_lines = []
        
        for table in schema.tables:
            cols = schema.columns.get(table, [])
            col_defs = [f"{c['name']} {c['type']}" for c in cols]
            schema_lines.append(f"CREATE TABLE {table} ({', '.join(col_defs)});")
        
        return "\n".join(schema_lines)
    
    def _extract_sql_from_text(self, text: str) -> str:
        """Extract SQL query from generated text"""
        # Look for SQL query patterns
        sql_pattern = r"(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER).*?;"
        matches = re.findall(sql_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Fallback: return everything after "SQL Query"
        if "SQL Query" in text:
            sql_part = text.split("SQL Query")[-1].strip()
            return sql_part.split("\n")[0].strip()
        
        return text.strip()
    
    def _generate_sql_rule_based(self, query: str, schema: DatabaseSchema) -> str:
        """Generate SQL using rule-based approach"""
        query_lower = query.lower()
        
        # Identify table
        table = None
        for t in schema.tables:
            if t.lower() in query_lower:
                table = t
                break
        
        if not table and schema.tables:
            table = schema.tables[0]  # Default to first table
        
        if not table:
            return ""
        
        # Build SELECT clause
        columns = schema.columns.get(table, [])
        col_names = [c['name'] for c in columns]
        
        # Determine which columns to select
        select_cols = "*"
        for col in col_names:
            if col.lower() in query_lower:
                select_cols = col
                break
        
        # Build WHERE clause
        where_clause = ""
        if "where" in query_lower or "with" in query_lower:
            # Simple pattern matching for conditions
            for col in col_names:
                if col.lower() in query_lower:
                    where_clause = f" WHERE {col} IS NOT NULL"
                    break
        
        # Build ORDER BY clause
        order_clause = ""
        if "latest" in query_lower or "recent" in query_lower:
            # Try to find date column
            date_cols = [c['name'] for c in columns if 'date' in c['name'].lower() or 'time' in c['name'].lower()]
            if date_cols:
                order_clause = f" ORDER BY {date_cols[0]} DESC"
        
        # Build LIMIT clause
        limit_clause = ""
        if "top" in query_lower or "first" in query_lower:
            # Extract number
            numbers = re.findall(r'\d+', query)
            if numbers:
                limit_clause = f" LIMIT {numbers[0]}"
            else:
                limit_clause = " LIMIT 10"
        
        sql = f"SELECT {select_cols} FROM {table}{where_clause}{order_clause}{limit_clause}"
        
        return sql
    
    def _is_query_safe(self, sql: str, mode: QueryMode) -> bool:
        """Validate query safety"""
        sql_upper = sql.upper()
        
        # Check for dangerous operations
        dangerous_ops = ["DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE"]
        
        if mode == QueryMode.READ_ONLY:
            if any(op in sql_upper for op in dangerous_ops):
                return False
            if "UPDATE" in sql_upper or "INSERT" in sql_upper:
                return False
        
        # Check for SQL injection patterns
        injection_patterns = [
            r";\s*(DROP|DELETE|UPDATE|INSERT)",
            r"--",
            r"/\*.*\*/",
            r"UNION\s+SELECT",
            r"exec\s*\(",
            r"execute\s*\("
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                return False
        
        return True
    
    def _execute_query(self, sql: str, connection_name: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        conn_info = self.connections.get(connection_name)
        
        if not conn_info:
            raise ValueError(f"Connection not found: {connection_name}")
        
        conn = conn_info["connection"]
        db_type = conn_info["type"]
        
        if db_type == DatabaseType.SQLITE:
            return pd.read_sql_query(sql, conn)
        elif db_type == DatabaseType.DUCKDB:
            return conn.execute(sql).fetchdf()
        elif SQLALCHEMY_AVAILABLE:
            return pd.read_sql_query(text(sql), conn)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def _generate_insights(self, df: pd.DataFrame, query: str) -> List[str]:
        """Generate insights from query results"""
        insights = []
        
        if df.empty:
            insights.append("No data found matching your query.")
            return insights
        
        # Basic statistics
        insights.append(f"Found {len(df)} rows")
        
        # Numeric column insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean_val = df[col].mean()
            median_val = df[col].median()
            insights.append(f"{col}: mean={mean_val:.2f}, median={median_val:.2f}")
            
            # Detect trends
            if len(df) > 2 and df[col].is_monotonic_increasing:
                insights.append(f"{col} shows an increasing trend")
            elif len(df) > 2 and df[col].is_monotonic_decreasing:
                insights.append(f"{col} shows a decreasing trend")
        
        # Categorical insights
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            unique_count = df[col].nunique()
            if unique_count < 10:
                top_value = df[col].value_counts().index[0]
                insights.append(f"Most common {col}: {top_value}")
        
        # Missing data
        missing = df.isnull().sum()
        if missing.any():
            missing_cols = missing[missing > 0]
            for col, count in missing_cols.items():
                pct = (count / len(df)) * 100
                insights.append(f"{col} has {pct:.1f}% missing values")
        
        return insights
    
    def _detect_seasonality(self, df: pd.DataFrame, col: str) -> bool:
        """Detect seasonality in time series"""
        # Simplified seasonality detection
        if len(df) < 12:
            return False
        
        try:
            values = df[col].values
            # Check for repeating patterns
            autocorr = np.correlate(values, values, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Look for peaks in autocorrelation
            peaks = (autocorr[1:-1] > autocorr[:-2]) & (autocorr[1:-1] > autocorr[2:])
            return np.sum(peaks) > 2
        except:
            return False
    
    def _detect_outliers(self, df: pd.DataFrame, col: str) -> List[Any]:
        """Detect outliers using IQR method"""
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            return outliers.tolist()
        except:
            return []
    
    def _create_visualization(self, df: pd.DataFrame, query: str) -> Optional[Any]:
        """Create visualization for query results"""
        if not PLOTLY_AVAILABLE or df.empty:
            return None
        
        try:
            # Determine best visualization type
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            cat_cols = df.select_dtypes(include=['object']).columns
            
            if len(numeric_cols) >= 2:
                # Scatter plot for numeric data
                fig = px.scatter(
                    df,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    title=f"{numeric_cols[0]} vs {numeric_cols[1]}"
                )
                return fig
            
            elif len(numeric_cols) == 1 and len(cat_cols) >= 1:
                # Bar chart
                fig = px.bar(
                    df,
                    x=cat_cols[0],
                    y=numeric_cols[0],
                    title=f"{numeric_cols[0]} by {cat_cols[0]}"
                )
                return fig
            
            elif len(numeric_cols) == 1:
                # Histogram
                fig = px.histogram(
                    df,
                    x=numeric_cols[0],
                    title=f"Distribution of {numeric_cols[0]}"
                )
                return fig
            
            else:
                # Table view
                fig = go.Figure(data=[go.Table(
                    header=dict(values=list(df.columns)),
                    cells=dict(values=[df[col] for col in df.columns])
                )])
                return fig
                
        except Exception as e:
            self.logger.error(f"Visualization creation failed: {e}")
            return None
    
    def get_schema_info(self, connection_name: str = "default") -> Optional[DatabaseSchema]:
        """Get schema information for a connection"""
        return self.schemas.get(connection_name)
    
    def close_connection(self, connection_name: str = "default"):
        """Close database connection"""
        if connection_name in self.connections:
            conn_info = self.connections[connection_name]
            conn = conn_info["connection"]
            
            try:
                if hasattr(conn, 'close'):
                    conn.close()
                elif hasattr(conn, 'dispose'):
                    conn.dispose()
                
                del self.connections[connection_name]
                if connection_name in self.schemas:
                    del self.schemas[connection_name]
                
                self.logger.info(f"Closed connection: {connection_name}")
            except Exception as e:
                self.logger.error(f"Error closing connection: {e}")
    
    def close_all_connections(self):
        """Close all database connections"""
        for name in list(self.connections.keys()):
            self.close_connection(name)
