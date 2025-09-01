import importlib
import pkgutil
import inspect
from typing import List, Dict, Type, Any
import logging
from langchain.llms.base import BaseLLM
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings

class IntegrationDiscovery:
    """Automatically discover LangChain integrations"""
    
    def __init__(self, patterns: List[str] = None):
        self.patterns = patterns or [
            "langchain_*",
            "langchain.*",
            "*langchain*"
        ]
        self.logger = logging.getLogger(__name__)
        
    def discover_all_integrations(self) -> Dict[str, List[Type]]:
        """Discover all LangChain integrations by type"""
        discovered = {
            "llms": [],
            "chat_models": [],
            "embeddings": []
        }
        
        # Discover from main langchain package
        discovered.update(self._discover_from_langchain())
        
        # Discover from langchain community packages
        discovered.update(self._discover_from_community_packages())
        
        # Discover from third-party packages
        discovered.update(self._discover_from_third_party())
        
        return discovered
    
    def _discover_from_langchain(self) -> Dict[str, List[Type]]:
        """Discover integrations from main langchain package"""
        integrations = {
            "llms": [],
            "chat_models": [], 
            "embeddings": []
        }
        
        try:
            # Discover LLMs
            from langchain import llms
            integrations["llms"].extend(self._find_classes_in_module(llms, BaseLLM))
            
            # Discover Chat Models
            from langchain import chat_models
            integrations["chat_models"].extend(self._find_classes_in_module(chat_models, BaseChatModel))
            
            # Discover Embeddings
            from langchain import embeddings
            integrations["embeddings"].extend(self._find_classes_in_module(embeddings, Embeddings))
            
        except ImportError as e:
            self.logger.warning(f"Could not import from langchain: {e}")
            
        return integrations
    
    def _discover_from_community_packages(self) -> Dict[str, List[Type]]:
        """Discover integrations from langchain-community package"""
        integrations = {
            "llms": [],
            "chat_models": [],
            "embeddings": []
        }
        
        try:
            # Try langchain-community package
            import langchain_community
            
            # Discover LLMs
            try:
                from langchain_community import llms
                integrations["llms"].extend(self._find_classes_in_module(llms, BaseLLM))
            except ImportError:
                pass
                
            # Discover Chat Models
            try:
                from langchain_community import chat_models
                integrations["chat_models"].extend(self._find_classes_in_module(chat_models, BaseChatModel))
            except ImportError:
                pass
                
            # Discover Embeddings
            try:
                from langchain_community import embeddings
                integrations["embeddings"].extend(self._find_classes_in_module(embeddings, Embeddings))
            except ImportError:
                pass
                
        except ImportError:
            self.logger.info("langchain-community package not available")
            
        return integrations
    
    def _discover_from_third_party(self) -> Dict[str, List[Type]]:
        """Discover integrations from third-party packages"""
        integrations = {
            "llms": [],
            "chat_models": [],
            "embeddings": []
        }
        
        # List of known third-party LangChain integration packages
        third_party_packages = [
            "langchain_openai",
            "langchain_anthropic", 
            "langchain_google_genai",
            "langchain_aws",
            "langchain_cohere",
            "langchain_huggingface",
            "langchain_mlx"
        ]
        
        for package_name in third_party_packages:
            try:
                package = importlib.import_module(package_name)
                
                # Try to find LLMs
                if hasattr(package, 'llms'):
                    integrations["llms"].extend(
                        self._find_classes_in_module(package.llms, BaseLLM)
                    )
                
                # Try to find Chat Models
                if hasattr(package, 'chat_models'):
                    integrations["chat_models"].extend(
                        self._find_classes_in_module(package.chat_models, BaseChatModel)
                    )
                
                # Try to find Embeddings
                if hasattr(package, 'embeddings'):
                    integrations["embeddings"].extend(
                        self._find_classes_in_module(package.embeddings, Embeddings)
                    )
                    
                # Also check root level for classes
                integrations["llms"].extend(self._find_classes_in_module(package, BaseLLM))
                integrations["chat_models"].extend(self._find_classes_in_module(package, BaseChatModel))
                integrations["embeddings"].extend(self._find_classes_in_module(package, Embeddings))
                
            except ImportError:
                self.logger.debug(f"Package {package_name} not available")
                continue
                
        return integrations
    
    def _find_classes_in_module(self, module: Any, base_class: Type) -> List[Type]:
        """Find all classes in a module that inherit from base_class"""
        classes = []
        
        try:
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, base_class) and 
                    obj != base_class and
                    not inspect.isabstract(obj)):
                    classes.append(obj)
                    
        except Exception as e:
            self.logger.warning(f"Error scanning module {module}: {e}")
            
        return classes
    
    def get_integration_info(self, integration_class: Type) -> Dict[str, Any]:
        """Get detailed information about an integration"""
        info = {
            "name": integration_class.__name__,
            "module": integration_class.__module__,
            "version": getattr(integration_class, "__version__", "unknown"),
            "doc": integration_class.__doc__ or "",
            "methods": [],
            "required_params": [],
            "optional_params": []
        }
        
        # Get method information
        for name, method in inspect.getmembers(integration_class, predicate=inspect.ismethod):
            if not name.startswith('_'):
                info["methods"].append(name)
        
        # Get constructor parameters
        try:
            sig = inspect.signature(integration_class.__init__)
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                    
                if param.default == inspect.Parameter.empty:
                    info["required_params"].append(param_name)
                else:
                    info["optional_params"].append(param_name)
                    
        except Exception as e:
            self.logger.warning(f"Could not inspect constructor for {integration_class.__name__}: {e}")
            
        return info