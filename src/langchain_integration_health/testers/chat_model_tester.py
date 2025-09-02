from typing import Dict, Any, List
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from .base_tester import BaseIntegrationTester

class ChatModelTester(BaseIntegrationTester):
    """Specialized tester for Chat Model integrations"""
    
    REQUIRED_METHODS = [
        'invoke', 'ainvoke', 'stream', 'astream',
        'bind_tools', 'with_structured_output'
    ]
    
    OPTIONAL_METHODS = [
        'batch', 'abatch', 'bind', 'with_config',
        'get_num_tokens', 'get_token_ids'
    ]
    
    async def _test_method_functionality(self) -> Dict[str, Any]:
        """Test actual method execution for Chat Models"""
        functionality_results = {}
        
        try:
            instance = self.integration_class(**self.config)
            
            # Test message handling
            functionality_results['message_handling'] = await self._test_message_handling(instance)
            
            # Test system message support
            functionality_results['system_message'] = await self._test_system_message_support(instance)
            
            # Test conversation handling
            functionality_results['conversation'] = await self._test_conversation_handling(instance)
            
            # Run base LLM tests
            base_results = await super()._test_method_functionality()
            functionality_results.update(base_results)
            
        except Exception as e:
            self.results.errors.append(f"Chat model functionality testing failed: {str(e)}")
            
        return functionality_results
    
    async def _test_message_handling(self, instance: Any) -> bool:
        """Test handling of different message types"""
        try:
            messages = [HumanMessage(content="Hello, how are you?")]
            response = instance.invoke(messages)
            
            if response and hasattr(response, 'content'):
                return True
            else:
                self.results.warnings.append("Message handling returned invalid response format")
                return False
                
        except Exception as e:
            self.results.errors.append(f"Message handling test failed: {str(e)}")
            return False
    
    async def _test_system_message_support(self, instance: Any) -> bool:
        """Test system message support"""
        try:
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="What is 2+2?")
            ]
            response = instance.invoke(messages)
            
            if response:
                return True
            else:
                self.results.warnings.append("System message support test returned no response")
                return False
                
        except Exception as e:
            self.results.warnings.append(f"System message support test failed: {str(e)}")
            return False
    
    async def _test_conversation_handling(self, instance: Any) -> bool:
        """Test multi-turn conversation handling"""
        try:
            messages = [
                HumanMessage(content="My name is Alice."),
                HumanMessage(content="What is my name?")
            ]
            response = instance.invoke(messages)
            
            if response:
                return True
            else:
                self.results.warnings.append("Conversation handling test returned no response")
                return False
                
        except Exception as e:
            self.results.warnings.append(f"Conversation handling test failed: {str(e)}")
            return False
    
    async def _test_specific_error_scenarios(self, instance: Any) -> Dict[str, bool]:
        """Test chat model specific error scenarios"""
        error_scenarios = {}
        
        # Test invalid message format
        try:
            instance.invoke("invalid_message_format")
            error_scenarios['handles_invalid_message_format'] = True
        except Exception:
            error_scenarios['handles_invalid_message_format'] = False
            
        # Test empty message list
        try:
            instance.invoke([])
            error_scenarios['handles_empty_messages'] = True
        except Exception:
            error_scenarios['handles_empty_messages'] = False
            
        # Test malformed messages
        try:
            invalid_messages = [{"invalid": "message"}]
            instance.invoke(invalid_messages)
            error_scenarios['handles_malformed_messages'] = True
        except Exception:
            error_scenarios['handles_malformed_messages'] = False
            
        return error_scenarios