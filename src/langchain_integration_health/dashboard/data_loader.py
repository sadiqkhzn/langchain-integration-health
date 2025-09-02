import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ..testers.base_tester import IntegrationTestResult

class DataLoader:
    """Handle loading and persisting test result data"""
    
    def __init__(self, db_path: str = "integration_health.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    integration_name TEXT NOT NULL,
                    integration_version TEXT NOT NULL,
                    test_timestamp TIMESTAMP NOT NULL,
                    bind_tools_support BOOLEAN NOT NULL,
                    streaming_support BOOLEAN NOT NULL,
                    structured_output_support BOOLEAN NOT NULL,
                    async_support BOOLEAN NOT NULL,
                    compatibility_score REAL NOT NULL,
                    errors TEXT,
                    warnings TEXT,
                    performance_metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_integration_name 
                ON test_results(integration_name)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_test_timestamp 
                ON test_results(test_timestamp)
            """)
    
    def save_test_result(self, result: IntegrationTestResult) -> None:
        """Save a test result to the database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO test_results (
                    integration_name, integration_version, test_timestamp,
                    bind_tools_support, streaming_support, structured_output_support,
                    async_support, compatibility_score, errors, warnings, performance_metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.integration_name,
                result.integration_version,
                result.test_timestamp,
                result.bind_tools_support,
                result.streaming_support,
                result.structured_output_support,
                result.async_support,
                result.compatibility_score,
                json.dumps(result.errors),
                json.dumps(result.warnings),
                json.dumps(result.performance_metrics)
            ))
    
    def load_test_results(self, limit: Optional[int] = None) -> List[IntegrationTestResult]:
        """Load test results from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = """
                SELECT * FROM test_results 
                ORDER BY test_timestamp DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor = conn.execute(query)
            rows = cursor.fetchall()
        
        results = []
        for row in rows:
            result = IntegrationTestResult(
                integration_name=row["integration_name"],
                integration_version=row["integration_version"],
                test_timestamp=datetime.fromisoformat(row["test_timestamp"]),
                bind_tools_support=bool(row["bind_tools_support"]),
                streaming_support=bool(row["streaming_support"]),
                structured_output_support=bool(row["structured_output_support"]),
                async_support=bool(row["async_support"]),
                errors=json.loads(row["errors"]) if row["errors"] else [],
                warnings=json.loads(row["warnings"]) if row["warnings"] else [],
                performance_metrics=json.loads(row["performance_metrics"]) if row["performance_metrics"] else {},
                compatibility_score=float(row["compatibility_score"])
            )
            results.append(result)
        
        return results
    
    def load_latest_results_by_integration(self) -> List[IntegrationTestResult]:
        """Load the most recent test result for each integration"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get the latest result for each integration
            cursor = conn.execute("""
                SELECT * FROM test_results t1
                WHERE t1.test_timestamp = (
                    SELECT MAX(t2.test_timestamp)
                    FROM test_results t2
                    WHERE t2.integration_name = t1.integration_name
                )
                ORDER BY t1.compatibility_score DESC
            """)
            
            rows = cursor.fetchall()
        
        results = []
        for row in rows:
            result = IntegrationTestResult(
                integration_name=row["integration_name"],
                integration_version=row["integration_version"],
                test_timestamp=datetime.fromisoformat(row["test_timestamp"]),
                bind_tools_support=bool(row["bind_tools_support"]),
                streaming_support=bool(row["streaming_support"]),
                structured_output_support=bool(row["structured_output_support"]),
                async_support=bool(row["async_support"]),
                errors=json.loads(row["errors"]) if row["errors"] else [],
                warnings=json.loads(row["warnings"]) if row["warnings"] else [],
                performance_metrics=json.loads(row["performance_metrics"]) if row["performance_metrics"] else {},
                compatibility_score=float(row["compatibility_score"])
            )
            results.append(result)
        
        return results
    
    def load_historical_data(self, days: int = 30) -> List[Dict[str, Any]]:
        """Load historical test data for trend analysis"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            cursor = conn.execute("""
                SELECT 
                    integration_name,
                    test_timestamp,
                    compatibility_score,
                    bind_tools_support,
                    streaming_support,
                    structured_output_support,
                    async_support
                FROM test_results 
                WHERE test_timestamp >= ?
                ORDER BY test_timestamp ASC
            """, (cutoff_date.isoformat(),))
            
            rows = cursor.fetchall()
        
        return [
            {
                "integration_name": row["integration_name"],
                "test_timestamp": pd.to_datetime(row["test_timestamp"]),
                "compatibility_score": row["compatibility_score"],
                "bind_tools_support": bool(row["bind_tools_support"]),
                "streaming_support": bool(row["streaming_support"]),
                "structured_output_support": bool(row["structured_output_support"]),
                "async_support": bool(row["async_support"])
            }
            for row in rows
        ]
    
    def get_integration_history(self, integration_name: str) -> List[IntegrationTestResult]:
        """Get test history for a specific integration"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("""
                SELECT * FROM test_results 
                WHERE integration_name = ?
                ORDER BY test_timestamp DESC
            """, (integration_name,))
            
            rows = cursor.fetchall()
        
        results = []
        for row in rows:
            result = IntegrationTestResult(
                integration_name=row["integration_name"],
                integration_version=row["integration_version"],
                test_timestamp=datetime.fromisoformat(row["test_timestamp"]),
                bind_tools_support=bool(row["bind_tools_support"]),
                streaming_support=bool(row["streaming_support"]),
                structured_output_support=bool(row["structured_output_support"]),
                async_support=bool(row["async_support"]),
                errors=json.loads(row["errors"]) if row["errors"] else [],
                warnings=json.loads(row["warnings"]) if row["warnings"] else [],
                performance_metrics=json.loads(row["performance_metrics"]) if row["performance_metrics"] else {},
                compatibility_score=float(row["compatibility_score"])
            )
            results.append(result)
        
        return results
    
    def delete_old_results(self, days: int = 90) -> int:
        """Delete test results older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM test_results 
                WHERE test_timestamp < ?
            """, (cutoff_date.isoformat(),))
            
            return cursor.rowcount