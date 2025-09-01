from typing import Dict, Any, List
import asyncio
import time
from langchain.schema import BaseMessage, HumanMessage
from langchain.tools import tool
from .base_tester import BaseIntegrationTester

class LLMIntegrationTester(BaseIntegrationTester):
    """Specialized tester for LLM integrations"""
    
    REQUIRED_METHODS = [
        'invoke', 'ainvoke', 'stream', 'astream',
        'bind_tools', 'with_structured_output'
    ]
    
    OPTIONAL_METHODS = [
        'batch', 'abatch', 'bind', 'with_config'
    ]
    
    async def _test_method_functionality(self) -> Dict[str, Any]:
        """Test actual method execution for LLM integrations"""
        functionality_results = {}
        
        try:
            instance = self.integration_class(**self.config)
            
            # Test basic invoke
            functionality_results['invoke'] = await self._test_invoke(instance)
            
            # Test async invoke
            functionality_results['ainvoke'] = await self._test_ainvoke(instance)
            
            # Test streaming
            functionality_results['streaming'] = await self._test_streaming_support(instance)
            
            # Test bind_tools
            functionality_results['bind_tools'] = await self._test_bind_tools_support(instance)
            
            # Test structured output
            functionality_results['structured_output'] = await self._test_structured_output(instance)
            
        except Exception as e:
            self.results.errors.append(f"Method functionality testing failed: {str(e)}")
            
        return functionality_results
    
    async def _test_invoke(self, instance: Any) -> bool:
        """Test basic invoke functionality"""
        try:
            start_time = time.time()
            response = instance.invoke("Hello, this is a test message.")
            end_time = time.time()
            
            self.results.performance_metrics['invoke_latency'] = end_time - start_time
            
            if response and len(str(response)) > 0:
                return True
            else:
                self.results.warnings.append("Invoke returned empty or invalid response")
                return False
                
        except Exception as e:
            self.results.errors.append(f"Invoke test failed: {str(e)}")
            return False
    
    async def _test_ainvoke(self, instance: Any) -> bool:
        """Test async invoke functionality"""
        try:
            if not hasattr(instance, 'ainvoke'):
                self.results.warnings.append("ainvoke method not available")
                return False
                
            start_time = time.time()
            response = await instance.ainvoke("Hello, this is an async test message.")
            end_time = time.time()
            
            self.results.performance_metrics['ainvoke_latency'] = end_time - start_time
            self.results.async_support = True
            
            if response and len(str(response)) > 0:
                return True
            else:
                self.results.warnings.append("Async invoke returned empty or invalid response")
                return False
                
        except Exception as e:
            self.results.errors.append(f"Async invoke test failed: {str(e)}")
            return False
    
    async def _test_streaming_support(self, instance: Any) -> bool:
        """Test streaming capabilities"""
        try:
            if not hasattr(instance, 'stream'):
                self.results.warnings.append("stream method not available")
                return False
                
            chunks = []
            start_time = time.time()
            
            async for chunk in instance.stream("Tell me a short story."):
                chunks.append(chunk)
                if len(chunks) > 10:  # Limit chunks for testing
                    break
                    
            end_time = time.time()
            
            if chunks:
                self.results.streaming_support = True
                self.results.performance_metrics['streaming_latency'] = end_time - start_time
                self.results.performance_metrics['chunks_received'] = len(chunks)
                return True
            else:
                self.results.warnings.append("Streaming produced no chunks")
                return False
                
        except Exception as e:
            self.results.errors.append(f"Streaming test failed: {str(e)}")
            return False
    
    async def _test_bind_tools_support(self, instance: Any) -> bool:
        """Test bind_tools functionality specifically"""
        try:
            if not hasattr(instance, 'bind_tools'):
                self.results.errors.append("bind_tools method not available")
                return False
            
            # Create a simple test tool
            @tool
            def test_calculator(a: int, b: int) -> int:
                """Add two numbers together."""
                return a + b
            
            # Attempt to bind the tool
            bound_model = instance.bind_tools([test_calculator])
            
            if bound_model:
                self.results.bind_tools_support = True
                
                # Test tool execution if possible
                try:
                    response = bound_model.invoke("What is 5 + 3?")
                    if response:
                        return True
                except Exception as tool_e:
                    self.results.warnings.append(f"Tool binding works but execution failed: {str(tool_e)}")
                    return True  # Binding itself works
                    
            return False
            
        except Exception as e:
            self.results.errors.append(f"bind_tools test failed: {str(e)}")
            return False
    
    async def _test_structured_output(self, instance: Any) -> bool:
        """Test with_structured_output functionality"""
        try:
            if not hasattr(instance, 'with_structured_output'):
                self.results.warnings.append("with_structured_output method not available")
                return False
            
            from pydantic import BaseModel
            
            class TestResponse(BaseModel):
                answer: str
                confidence: float
            
            # Attempt to use structured output
            structured_model = instance.with_structured_output(TestResponse)
            
            if structured_model:
                self.results.structured_output_support = True
                
                # Test structured output execution
                try:
                    response = structured_model.invoke("Respond with answer='test' and confidence=0.9")
                    if isinstance(response, TestResponse):
                        return True
                except Exception as struct_e:
                    self.results.warnings.append(f"Structured output binding works but execution failed: {str(struct_e)}")
                    return True  # Binding itself works
                    
            return False
            
        except Exception as e:
            self.results.errors.append(f"Structured output test failed: {str(e)}")
            return False
    
    async def _test_specific_error_scenarios(self, instance: Any) -> Dict[str, bool]:
        """Test LLM-specific error scenarios"""
        error_scenarios = {}
        
        # Test invalid input handling
        try:
            instance.invoke("")  # Empty input
            error_scenarios['handles_empty_input'] = True
        except Exception:
            error_scenarios['handles_empty_input'] = False
            
        # Test very long input handling
        try:
            long_input = "test " * 10000
            instance.invoke(long_input)
            error_scenarios['handles_long_input'] = True
        except Exception:
            error_scenarios['handles_long_input'] = False
            
        # Test invalid tool binding
        try:
            if hasattr(instance, 'bind_tools'):
                instance.bind_tools([])  # Empty tools list
                error_scenarios['handles_empty_tools'] = True
        except Exception:
            error_scenarios['handles_empty_tools'] = False
            
        return error_scenarios