from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from datetime import datetime
from dataclasses import dataclass, field
import asyncio
import logging
import inspect
import traceback

@dataclass
class IntegrationTestResult:
    integration_name: str
    integration_version: str
    test_timestamp: datetime
    bind_tools_support: bool
    streaming_support: bool
    structured_output_support: bool
    async_support: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    compatibility_score: float = 0.0

class BaseIntegrationTester(ABC):
    """Base class for testing LangChain integrations"""
    
    REQUIRED_METHODS: List[str] = []
    OPTIONAL_METHODS: List[str] = []
    
    def __init__(self, integration_class: Type, config: Optional[Dict[str, Any]] = None):
        self.integration_class = integration_class
        self.config = config or {}
        self.results = IntegrationTestResult(
            integration_name=integration_class.__name__,
            integration_version=getattr(integration_class, "__version__", "unknown"),
            test_timestamp=datetime.now(),
            bind_tools_support=False,
            streaming_support=False,
            structured_output_support=False,
            async_support=False
        )
        self.logger = logging.getLogger(f"{__name__}.{integration_class.__name__}")
        
    async def run_all_tests(self) -> IntegrationTestResult:
        """Run comprehensive integration tests"""
        try:
            # Test basic instantiation
            await self._test_instantiation()
            
            # Test required methods existence
            self._test_required_methods()
            
            # Test method functionality
            await self._test_method_functionality()
            
            # Test error handling
            await self._test_error_handling()
            
            # Calculate compatibility score
            self._calculate_compatibility_score()
            
        except Exception as e:
            self.results.errors.append(f"Test execution failed: {str(e)}")
            self.logger.error(f"Test execution failed for {self.integration_class.__name__}: {e}")
            
        return self.results
    
    async def _test_instantiation(self) -> bool:
        """Test if integration can be instantiated"""
        try:
            # Try to instantiate with minimal config
            instance = self.integration_class(**self.config)
            self.logger.info(f"Successfully instantiated {self.integration_class.__name__}")
            return True
        except Exception as e:
            self.results.errors.append(f"Instantiation failed: {str(e)}")
            return False
    
    def _test_required_methods(self) -> Dict[str, bool]:
        """Test if integration has required methods"""
        method_results = {}
        
        for method_name in self.REQUIRED_METHODS:
            has_method = hasattr(self.integration_class, method_name)
            method_results[method_name] = has_method
            
            if not has_method:
                self.results.errors.append(f"Missing required method: {method_name}")
            else:
                # Check if method is callable
                method = getattr(self.integration_class, method_name)
                if not callable(method):
                    self.results.errors.append(f"Method {method_name} is not callable")
                    method_results[method_name] = False
                    
        return method_results
    
    @abstractmethod
    async def _test_method_functionality(self) -> Dict[str, Any]:
        """Test actual method execution - must be implemented by subclasses"""
        pass
    
    async def _test_error_handling(self) -> Dict[str, bool]:
        """Test error handling capabilities"""
        error_handling_results = {}
        
        try:
            # Test with invalid inputs
            instance = self.integration_class(**self.config)
            
            # Test various error scenarios specific to integration type
            error_handling_results = await self._test_specific_error_scenarios(instance)
            
        except Exception as e:
            self.results.warnings.append(f"Error handling test failed: {str(e)}")
            
        return error_handling_results
    
    @abstractmethod
    async def _test_specific_error_scenarios(self, instance: Any) -> Dict[str, bool]:
        """Test integration-specific error scenarios"""
        pass
    
    def _calculate_compatibility_score(self) -> float:
        """Calculate overall compatibility score based on test results"""
        total_features = 4  # bind_tools, streaming, structured_output, async
        supported_features = sum([
            self.results.bind_tools_support,
            self.results.streaming_support, 
            self.results.structured_output_support,
            self.results.async_support
        ])
        
        # Base score from feature support
        feature_score = (supported_features / total_features) * 0.7
        
        # Penalty for errors
        error_penalty = min(len(self.results.errors) * 0.1, 0.3)
        
        # Bonus for no warnings
        warning_penalty = min(len(self.results.warnings) * 0.05, 0.2)
        
        self.results.compatibility_score = max(0.0, feature_score - error_penalty - warning_penalty)
        return self.results.compatibility_score
    
    def _check_async_support(self, method_name: str) -> bool:
        """Check if a method has async support"""
        if not hasattr(self.integration_class, method_name):
            return False
            
        method = getattr(self.integration_class, method_name)
        return asyncio.iscoroutinefunction(method)