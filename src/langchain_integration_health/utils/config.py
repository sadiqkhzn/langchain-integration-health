from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import os
import json

class Config(BaseModel):
    """Configuration management for integration testing"""
    
    # Database settings
    database_url: str = Field(default="sqlite:///integration_health.db")
    
    # Testing settings
    test_timeout: int = Field(default=30, description="Test timeout in seconds")
    parallel_tests: bool = Field(default=True, description="Run tests in parallel")
    mock_mode: bool = Field(default=False, description="Run in mock mode without API calls")
    
    # Dashboard settings
    dashboard_host: str = Field(default="localhost")
    dashboard_port: int = Field(default=8501)
    
    # Integration discovery settings
    auto_discovery: bool = Field(default=True, description="Automatically discover integrations")
    discovery_patterns: list[str] = Field(default_factory=lambda: [
        "langchain_*",
        "langchain.*", 
        "*langchain*"
    ])
    
    # API keys for testing (optional)
    api_keys: Dict[str, str] = Field(default_factory=dict)
    
    # Performance settings
    performance_tracking: bool = Field(default=True)
    benchmark_iterations: int = Field(default=3)
    
    @classmethod
    def from_file(cls, file_path: str) -> "Config":
        """Load configuration from JSON file"""
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        return cls()
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        config_data = {}
        
        # Map environment variables to config fields
        env_mapping = {
            "LIH_DATABASE_URL": "database_url",
            "LIH_TEST_TIMEOUT": "test_timeout", 
            "LIH_PARALLEL_TESTS": "parallel_tests",
            "LIH_MOCK_MODE": "mock_mode",
            "LIH_DASHBOARD_HOST": "dashboard_host",
            "LIH_DASHBOARD_PORT": "dashboard_port",
            "LIH_AUTO_DISCOVERY": "auto_discovery",
            "LIH_PERFORMANCE_TRACKING": "performance_tracking",
            "LIH_BENCHMARK_ITERATIONS": "benchmark_iterations"
        }
        
        for env_var, config_field in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Convert string values to appropriate types
                if config_field in ["test_timeout", "dashboard_port", "benchmark_iterations"]:
                    config_data[config_field] = int(value)
                elif config_field in ["parallel_tests", "mock_mode", "auto_discovery", "performance_tracking"]:
                    config_data[config_field] = value.lower() in ("true", "1", "yes")
                else:
                    config_data[config_field] = value
        
        # Load API keys from environment
        api_keys = {}
        for key, value in os.environ.items():
            if key.endswith("_API_KEY"):
                provider_name = key[:-8].lower()  # Remove _API_KEY suffix
                api_keys[provider_name] = value
        
        if api_keys:
            config_data["api_keys"] = api_keys
            
        return cls(**config_data)
    
    def to_file(self, file_path: str) -> None:
        """Save configuration to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
    
    def get_integration_config(self, integration_name: str) -> Dict[str, Any]:
        """Get configuration specific to an integration"""
        integration_config = {}
        
        # Add API key if available
        provider_name = self._extract_provider_name(integration_name)
        if provider_name in self.api_keys:
            integration_config["api_key"] = self.api_keys[provider_name]
            
        # Add common settings
        integration_config.update({
            "timeout": self.test_timeout,
            "mock_mode": self.mock_mode
        })
        
        return integration_config
    
    def _extract_provider_name(self, integration_name: str) -> str:
        """Extract provider name from integration class name"""
        # Common patterns for provider extraction
        name_lower = integration_name.lower()
        
        if "openai" in name_lower:
            return "openai"
        elif "anthropic" in name_lower or "claude" in name_lower:
            return "anthropic"
        elif "google" in name_lower or "gemini" in name_lower:
            return "google"
        elif "azure" in name_lower:
            return "azure"
        elif "aws" in name_lower or "bedrock" in name_lower:
            return "aws"
        elif "huggingface" in name_lower or "hf" in name_lower:
            return "huggingface"
        elif "cohere" in name_lower:
            return "cohere"
        elif "mlx" in name_lower:
            return "mlx"
        else:
            return integration_name.lower()