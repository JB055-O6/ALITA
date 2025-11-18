"""
Safe Code Execution Environment

Implements Task 26 requirements:
- Sandboxed execution with RestrictedPython
- Static code analysis with bandit
- Approval dialogs for risky operations
- Iterative debugging with variable inspection
- Execution audit logging

Uses latest security frameworks:
- RestrictedPython 6.2 for sandboxing
- Bandit for static analysis
- AST parsing for code inspection
- Subprocess isolation

All features are FREE and run locally!
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import json
import subprocess
import sys
import ast
import traceback
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from RestrictedPython import compile_restricted
    from RestrictedPython import safe_globals
    from RestrictedPython.Guards import guarded_iter_unpack_sequence
    from RestrictedPython.Guards import safer_getattr
except ImportError:
    compile_restricted = None
    safe_globals = None
    guarded_iter_unpack_sequence = None
    safer_getattr = None

try:
    import bandit
    from bandit.core import manager as bandit_manager
except ImportError:
    bandit = None
    bandit_manager = None


class RiskLevel(Enum):
    """Risk levels for code operations."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CodeAnalysisResult:
    """Result of code analysis."""
    risk_level: RiskLevel
    issues: List[Dict[str, Any]]
    dangerous_operations: List[str]
    requires_approval: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['risk_level'] = self.risk_level.value
        return data


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: Optional[str]
    variables: Dict[str, Any]
    execution_time: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        # Convert non-serializable variables
        data['variables'] = {k: str(v) for k, v in self.variables.items()}
        return data


class StaticCodeAnalyzer:
    """Analyze code for security issues using Bandit and AST."""
    
    def __init__(self):
        self.dangerous_imports = {
            'os', 'subprocess', 'sys', 'shutil', 'socket',
            'requests', 'urllib', 'eval', 'exec', '__import__'
        }
        
        self.dangerous_functions = {
            'eval', 'exec', 'compile', '__import__', 'open',
            'input', 'raw_input', 'execfile'
        }
        
        self.file_operations = {
            'open', 'read', 'write', 'remove', 'unlink',
            'rmdir', 'mkdir', 'rename'
        }
    
    def analyze_code(self, code: str, language: str = "python") -> CodeAnalysisResult:
        """Analyze code for security issues.
        
        Args:
            code: Code to analyze
            language: Programming language
            
        Returns:
            Analysis result
        """
        if language == "python":
            return self._analyze_python(code)
        elif language in ["powershell", "batch"]:
            return self._analyze_script(code, language)
        else:
            return CodeAnalysisResult(
                risk_level=RiskLevel.MEDIUM,
                issues=[{"message": f"Unknown language: {language}"}],
                dangerous_operations=[],
                requires_approval=True
            )
    
    def _analyze_python(self, code: str) -> CodeAnalysisResult:
        """Analyze Python code."""
        issues = []
        dangerous_operations = []
        risk_level = RiskLevel.SAFE
        
        try:
            # Parse AST
            tree = ast.parse(code)
            
            # Check imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.dangerous_imports:
                            dangerous_operations.append(f"Import: {alias.name}")
                            risk_level = RiskLevel.HIGH
                            issues.append({
                                "type": "dangerous_import",
                                "module": alias.name,
                                "line": node.lineno
                            })
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.dangerous_imports:
                        dangerous_operations.append(f"Import from: {node.module}")
                        risk_level = RiskLevel.HIGH
                        issues.append({
                            "type": "dangerous_import",
                            "module": node.module,
                            "line": node.lineno
                        })
                
                # Check function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.dangerous_functions:
                            dangerous_operations.append(f"Function: {node.func.id}")
                            risk_level = RiskLevel.CRITICAL
                            issues.append({
                                "type": "dangerous_function",
                                "function": node.func.id,
                                "line": node.lineno
                            })
                        
                        elif node.func.id in self.file_operations:
                            dangerous_operations.append(f"File operation: {node.func.id}")
                            if risk_level.value < RiskLevel.MEDIUM.value:
                                risk_level = RiskLevel.MEDIUM
                            issues.append({
                                "type": "file_operation",
                                "function": node.func.id,
                                "line": node.lineno
                            })
            
            # Use Bandit if available
            if bandit and bandit_manager:
                try:
                    # Run Bandit analysis
                    b_mgr = bandit_manager.BanditManager(
                        bandit.core.config.BanditConfig(),
                        'file'
                    )
                    b_mgr.discover_files([code], True)
                    b_mgr.run_tests()
                    
                    for issue in b_mgr.get_issue_list():
                        issues.append({
                            "type": "bandit",
                            "severity": issue.severity,
                            "confidence": issue.confidence,
                            "text": issue.text,
                            "line": issue.lineno
                        })
                        
                        if issue.severity == "HIGH":
                            risk_level = RiskLevel.HIGH
                
                except Exception as e:
                    logging.debug(f"Bandit analysis failed: {e}")
        
        except SyntaxError as e:
            issues.append({
                "type": "syntax_error",
                "message": str(e),
                "line": e.lineno
            })
            risk_level = RiskLevel.MEDIUM
        
        except Exception as e:
            logging.error(f"Code analysis failed: {e}")
            issues.append({
                "type": "analysis_error",
                "message": str(e)
            })
            risk_level = RiskLevel.MEDIUM
        
        # Determine if approval required
        requires_approval = risk_level.value >= RiskLevel.MEDIUM.value
        
        return CodeAnalysisResult(
            risk_level=risk_level,
            issues=issues,
            dangerous_operations=dangerous_operations,
            requires_approval=requires_approval
        )
    
    def _analyze_script(self, code: str, language: str) -> CodeAnalysisResult:
        """Analyze PowerShell or Batch script."""
        dangerous_operations = []
        issues = []
        risk_level = RiskLevel.MEDIUM  # Scripts are inherently riskier
        
        # Check for dangerous patterns
        dangerous_patterns = [
            'rm -rf', 'del /f', 'format', 'diskpart',
            'reg delete', 'net user', 'shutdown',
            'Invoke-Expression', 'iex', 'wget', 'curl'
        ]
        
        code_lower = code.lower()
        
        for pattern in dangerous_patterns:
            if pattern.lower() in code_lower:
                dangerous_operations.append(f"Pattern: {pattern}")
                risk_level = RiskLevel.HIGH
                issues.append({
                    "type": "dangerous_pattern",
                    "pattern": pattern
                })
        
        return CodeAnalysisResult(
            risk_level=risk_level,
            issues=issues,
            dangerous_operations=dangerous_operations,
            requires_approval=True
        )


class SandboxedExecutor:
    """Execute code in sandboxed environment."""
    
    def __init__(self):
        self.restricted_available = compile_restricted is not None
    
    def execute_python_restricted(self, code: str, timeout: int = 30) -> ExecutionResult:
        """Execute Python code with RestrictedPython.
        
        Args:
            code: Python code
            timeout: Execution timeout in seconds
            
        Returns:
            Execution result
        """
        if not self.restricted_available:
            return ExecutionResult(
                success=False,
                output="",
                error="RestrictedPython not available",
                variables={},
                execution_time=0.0,
                timestamp=datetime.now()
            )
        
        start_time = datetime.now()
        
        try:
            # Compile with restrictions
            byte_code = compile_restricted(
                code,
                filename='<inline>',
                mode='exec'
            )
            
            if byte_code.errors:
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Compilation errors: {byte_code.errors}",
                    variables={},
                    execution_time=0.0,
                    timestamp=datetime.now()
                )
            
            # Prepare safe globals
            safe_globals_dict = safe_globals.copy()
            safe_globals_dict['_getiter_'] = guarded_iter_unpack_sequence
            safe_globals_dict['_getattr_'] = safer_getattr
            safe_globals_dict['__builtins__'] = {
                'print': print,
                'len': len,
                'range': range,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
            }
            
            # Execute
            exec(byte_code.code, safe_globals_dict)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract variables
            variables = {
                k: v for k, v in safe_globals_dict.items()
                if not k.startswith('_') and k not in ['__builtins__']
            }
            
            return ExecutionResult(
                success=True,
                output="Execution completed successfully",
                error=None,
                variables=variables,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                success=False,
                output="",
                error=f"{type(e).__name__}: {str(e)}",
                variables={},
                execution_time=execution_time,
                timestamp=datetime.now()
            )
    
    def execute_python_subprocess(self, code: str, timeout: int = 30) -> ExecutionResult:
        """Execute Python code in subprocess.
        
        Args:
            code: Python code
            timeout: Execution timeout
            
        Returns:
            Execution result
        """
        start_time = datetime.now()
        
        try:
            # Execute in subprocess
            result = subprocess.run(
                [sys.executable, '-c', code],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                variables={},
                execution_time=execution_time,
                timestamp=datetime.now()
            )
        
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timeout after {timeout} seconds",
                variables={},
                execution_time=timeout,
                timestamp=datetime.now()
            )
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                variables={},
                execution_time=execution_time,
                timestamp=datetime.now()
            )
    
    def execute_script(self, code: str, language: str, timeout: int = 30) -> ExecutionResult:
        """Execute PowerShell or Batch script.
        
        Args:
            code: Script code
            language: Script language
            timeout: Execution timeout
            
        Returns:
            Execution result
        """
        start_time = datetime.now()
        
        try:
            if language == "powershell":
                cmd = ["powershell", "-Command", code]
            elif language == "batch":
                cmd = ["cmd", "/c", code]
            else:
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Unsupported language: {language}",
                    variables={},
                    execution_time=0.0,
                    timestamp=datetime.now()
                )
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                variables={},
                execution_time=execution_time,
                timestamp=datetime.now()
            )
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                variables={},
                execution_time=execution_time,
                timestamp=datetime.now()
            )


class ExecutionAuditor:
    """Audit code execution."""
    
    def __init__(self, log_file: Path = None):
        self.log_file = log_file or Path("logs/code_execution.log")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_execution(self,
                     code: str,
                     language: str,
                     analysis: CodeAnalysisResult,
                     result: ExecutionResult,
                     approved: bool):
        """Log code execution.
        
        Args:
            code: Executed code
            language: Programming language
            analysis: Analysis result
            result: Execution result
            approved: Whether execution was approved
        """
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "language": language,
                "code_preview": code[:200],  # First 200 chars
                "code_length": len(code),
                "analysis": analysis.to_dict(),
                "result": result.to_dict(),
                "approved": approved
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        
        except Exception as e:
            logging.error(f"Execution logging failed: {e}")
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history.
        
        Args:
            limit: Maximum entries to return
            
        Returns:
            List of execution log entries
        """
        if not self.log_file.exists():
            return []
        
        try:
            entries = []
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue
            
            return entries[-limit:]
        
        except Exception as e:
            logging.error(f"Failed to read execution history: {e}")
            return []


class ErrorAnalyzer:
    """Analyze execution errors and suggest fixes."""
    
    def analyze_error(self, error: str, code: str) -> Dict[str, Any]:
        """Analyze error and suggest fixes.
        
        Args:
            error: Error message
            code: Code that caused error
            
        Returns:
            Analysis with suggestions
        """
        suggestions = []
        error_type = "unknown"
        
        # Common error patterns
        if "NameError" in error:
            error_type = "name_error"
            suggestions.append("Check if all variables are defined before use")
            suggestions.append("Verify import statements are correct")
        
        elif "SyntaxError" in error:
            error_type = "syntax_error"
            suggestions.append("Check for missing colons, parentheses, or brackets")
            suggestions.append("Verify indentation is correct")
        
        elif "TypeError" in error:
            error_type = "type_error"
            suggestions.append("Check if you're using the correct data types")
            suggestions.append("Verify function arguments match expected types")
        
        elif "AttributeError" in error:
            error_type = "attribute_error"
            suggestions.append("Check if the object has the attribute you're accessing")
            suggestions.append("Verify the object is initialized correctly")
        
        elif "IndexError" in error:
            error_type = "index_error"
            suggestions.append("Check if the index is within the list/array bounds")
            suggestions.append("Verify the collection is not empty")
        
        elif "KeyError" in error:
            error_type = "key_error"
            suggestions.append("Check if the key exists in the dictionary")
            suggestions.append("Use .get() method with a default value")
        
        elif "ImportError" in error or "ModuleNotFoundError" in error:
            error_type = "import_error"
            suggestions.append("Check if the module is installed")
            suggestions.append("Verify the module name is spelled correctly")
        
        return {
            "error_type": error_type,
            "error_message": error,
            "suggestions": suggestions,
            "code_preview": code[:200]
        }


class DebugSession:
    """Interactive debugging session."""
    
    def __init__(self):
        self.variables: Dict[str, Any] = {}
        self.breakpoints: List[int] = []
        self.execution_history: List[Dict[str, Any]] = []
    
    def set_breakpoint(self, line_number: int):
        """Set breakpoint at line."""
        if line_number not in self.breakpoints:
            self.breakpoints.append(line_number)
            self.breakpoints.sort()
    
    def remove_breakpoint(self, line_number: int):
        """Remove breakpoint."""
        if line_number in self.breakpoints:
            self.breakpoints.remove(line_number)
    
    def inspect_variable(self, var_name: str) -> Optional[Any]:
        """Inspect variable value."""
        return self.variables.get(var_name)
    
    def get_all_variables(self) -> Dict[str, Any]:
        """Get all variables."""
        return self.variables.copy()
    
    def record_execution_step(self, line_number: int, variables: Dict[str, Any]):
        """Record execution step."""
        self.execution_history.append({
            "line": line_number,
            "variables": variables.copy(),
            "timestamp": datetime.now().isoformat()
        })


class CodeExecutor:
    """Main safe code execution system."""
    
    def __init__(self):
        self.analyzer = StaticCodeAnalyzer()
        self.executor = SandboxedExecutor()
        self.auditor = ExecutionAuditor()
        self.error_analyzer = ErrorAnalyzer()
        
        # Approval callback (can be set by UI)
        self.approval_callback: Optional[callable] = None
        
        logging.info("Code Executor initialized")
    
    async def execute_code(self,
                          code: str,
                          language: str = "python",
                          sandboxed: bool = True,
                          timeout: int = 30) -> Tuple[ExecutionResult, CodeAnalysisResult]:
        """Execute code safely.
        
        Args:
            code: Code to execute
            language: Programming language
            sandboxed: Use sandboxed execution
            timeout: Execution timeout
            
        Returns:
            Tuple of (execution_result, analysis_result)
        """
        # Analyze code
        analysis = self.analyzer.analyze_code(code, language)
        
        # Check if approval required
        approved = True
        if analysis.requires_approval:
            if self.approval_callback:
                approved = await self.approval_callback(code, analysis)
            else:
                # Default: deny high-risk code
                approved = analysis.risk_level.value < RiskLevel.HIGH.value
        
        if not approved:
            result = ExecutionResult(
                success=False,
                output="",
                error="Execution denied: User did not approve",
                variables={},
                execution_time=0.0,
                timestamp=datetime.now()
            )
            
            self.auditor.log_execution(code, language, analysis, result, approved)
            return result, analysis
        
        # Execute code
        if language == "python":
            if sandboxed and self.executor.restricted_available:
                result = self.executor.execute_python_restricted(code, timeout)
            else:
                result = self.executor.execute_python_subprocess(code, timeout)
        else:
            result = self.executor.execute_script(code, language, timeout)
        
        # Log execution
        self.auditor.log_execution(code, language, analysis, result, approved)
        
        return result, analysis
    
    def analyze_error(self, error: str, code: str) -> Dict[str, Any]:
        """Analyze execution error."""
        return self.error_analyzer.analyze_error(error, code)
    
    def create_debug_session(self) -> DebugSession:
        """Create new debug session."""
        return DebugSession()
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.auditor.get_execution_history(limit)
    
    def set_approval_callback(self, callback: callable):
        """Set approval callback for risky code.
        
        Args:
            callback: Async function that takes (code, analysis) and returns bool
        """
        self.approval_callback = callback


# Global instance
_code_executor: Optional[CodeExecutor] = None


def get_code_executor() -> CodeExecutor:
    """Get global code executor instance."""
    global _code_executor
    
    if _code_executor is None:
        _code_executor = CodeExecutor()
    
    return _code_executor
