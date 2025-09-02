# LangChain Integration Health Dashboard & Testing Framework

A comprehensive testing framework and dashboard for monitoring LangChain integration compatibility, addressing critical pain points in the LangChain ecosystem.

## Problem Statement

LangChain has major integration compatibility issues:
- Missing `.bind_tools()` methods in integrations like MLXPipeline
- Inconsistent API support across different model providers
- Poor integration testing leading to breaking changes
- No visibility into which integrations support which features
- Developers waste time discovering integration limitations at runtime

## Solution

This framework provides:

1. **Integration Compatibility Testing Framework** - Automated testing suite that validates LangChain integrations
2. **Real-time Integration Health Dashboard** - Web-based dashboard showing integration compatibility status
3. **Integration Template Generator** - Standardized templates for new integrations

## Quick Start

### Installation

#### From PyPI (Recommended)

```bash
pip install langchain-integration-health
```

#### From Source

```bash
git clone https://github.com/sadiqkhzn/langchain-integration-health.git
cd langchain-integration-health
pip install -e .
```

### Run the Dashboard

```bash
langchain-health dashboard
```

Or run Streamlit directly:

```bash
streamlit run -m langchain_integration_health.dashboard.app
```

### CLI Usage

```bash
# Discover available integrations
langchain-health discover

# Run integration tests
langchain-health test

# Launch the dashboard
langchain-health dashboard

# Generate compatibility report
langchain-health report

# Clean old test results
langchain-health clean
```

### Programmatic Usage

```python
import asyncio
from langchain_integration_health.testers import LLMIntegrationTester
from langchain_integration_health.utils.config import Config

# Test a specific integration
async def test_integration():
    config = Config.from_env()
    
    # Example: Test OpenAI integration
    from langchain_openai import ChatOpenAI
    
    tester = LLMIntegrationTester(ChatOpenAI, config.get_integration_config("ChatOpenAI"))
    result = await tester.run_all_tests()
    
    print(f"Compatibility Score: {result.compatibility_score}")
    print(f"bind_tools Support: {result.bind_tools_support}")
    print(f"Streaming Support: {result.streaming_support}")

asyncio.run(test_integration())
```

## Features

### Comprehensive Testing
- Tests for required methods: `bind_tools()`, `stream()`, `with_structured_output()`, etc.
- Compatibility matrix generation
- Performance benchmarking
- Error handling validation

### Real-time Dashboard
- Visual compatibility matrix with color-coded status indicators
- Detailed test results and error reporting
- Performance metrics and benchmarking data
- Historical trend analysis
- Export capabilities (JSON, CSV, Markdown)

### Automatic Discovery
- Scans installed packages for LangChain integrations
- Supports main langchain, langchain-community, and third-party packages
- Parallel testing for faster results

### Integration Fixes
- Example implementations for common issues (e.g., MLXPipeline bind_tools fix)
- Wrapper patterns for adding missing functionality
- Best practices for LangChain compatibility

## Architecture

```
langchain-integration-health/
├── src/
│   ├── testers/           # Testing framework
│   │   ├── base_tester.py
│   │   ├── llm_tester.py
│   │   ├── chat_model_tester.py
│   │   └── embeddings_tester.py
│   ├── dashboard/         # Streamlit dashboard
│   │   ├── app.py
│   │   ├── components.py
│   │   └── data_loader.py
│   ├── utils/             # Utilities
│   │   ├── config.py
│   │   ├── reporters.py
│   │   └── discovery.py
│   └── examples/          # Example implementations
│       └── mlx_pipeline_fix.py
├── .github/workflows/     # CI/CD integration
└── tests/                 # Test suite
```

## Testing Framework

### Base Classes

- `BaseIntegrationTester`: Abstract base for all integration testers
- `LLMIntegrationTester`: Specialized for LLM integrations
- `ChatModelTester`: Specialized for chat models
- `EmbeddingsTester`: Specialized for embedding models

### Test Types

1. **Method Existence Tests**: Verify required methods are present
2. **Functionality Tests**: Test actual method execution
3. **Error Handling Tests**: Validate proper error handling
4. **Performance Tests**: Measure latency and throughput
5. **Compatibility Tests**: Check version compatibility

## Dashboard Features

### Compatibility Matrix
Color-coded grid showing which integrations support which features:
- Green: High compatibility (>=0.8)
- Yellow: Medium compatibility (0.5-0.8)
- Red: Low compatibility (<0.5)

### Integration Details
Expandable sections showing:
- Full test results
- Error and warning details
- Performance metrics
- Historical data

### Export Options
- JSON: Structured data for programmatic use
- CSV: Spreadsheet-compatible format
- Markdown: Human-readable reports

## Configuration

### Environment Variables

```bash
# Database
LIH_DATABASE_URL=sqlite:///integration_health.db

# Testing
LIH_TEST_TIMEOUT=30
LIH_PARALLEL_TESTS=true
LIH_MOCK_MODE=false

# Dashboard
LIH_DASHBOARD_HOST=localhost
LIH_DASHBOARD_PORT=8501

# API Keys (optional, for real testing)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

### Configuration File

```json
{
  "database_url": "sqlite:///integration_health.db",
  "test_timeout": 30,
  "parallel_tests": true,
  "mock_mode": false,
  "api_keys": {
    "openai": "your_key_here",
    "anthropic": "your_key_here"
  }
}
```

## CI/CD Integration

The framework includes GitHub Actions workflow for automated testing:

```yaml
# .github/workflows/integration-tests.yml
# Runs daily and on PRs to test all integrations
```

### Features:
- Automatic integration discovery
- Parallel testing across integrations
- Report generation and artifact upload
- PR comment integration

## MLXPipeline Fix Example

The framework includes a complete example of how to fix the MLXPipeline `bind_tools` issue:

```python
from langchain_integration_health.examples.mlx_pipeline_fix import create_mlx_wrapper

# Original MLXPipeline (missing bind_tools)
mlx = MLXPipeline(model_name="mlx-community/Llama-3.2-1B-Instruct-4bit")

# Wrap for LangChain compatibility
langchain_mlx = create_mlx_wrapper(mlx)

# Now you can use bind_tools!
from langchain.tools import tool

@tool
def calculator(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

mlx_with_tools = langchain_mlx.bind_tools([calculator])
response = mlx_with_tools.invoke("What is 5 + 3?")
```

## API Reference

### Testing Classes

#### `BaseIntegrationTester`
Base class for all integration testers.

```python
class BaseIntegrationTester:
    def __init__(self, integration_class: Type, config: Optional[Dict] = None)
    async def run_all_tests(self) -> IntegrationTestResult
    def _test_required_methods(self) -> Dict[str, bool]
```

#### `LLMIntegrationTester`
Specialized tester for LLM integrations.

```python
class LLMIntegrationTester(BaseIntegrationTester):
    REQUIRED_METHODS = ['invoke', 'ainvoke', 'stream', 'astream', 'bind_tools', 'with_structured_output']
    
    async def _test_bind_tools_support(self) -> bool
    async def _test_streaming_support(self) -> bool
```

### Data Models

#### `IntegrationTestResult`
Stores test results for an integration.

```python
@dataclass
class IntegrationTestResult:
    integration_name: str
    integration_version: str
    test_timestamp: datetime
    bind_tools_support: bool
    streaming_support: bool
    structured_output_support: bool
    async_support: bool
    errors: List[str]
    warnings: List[str]
    performance_metrics: Dict[str, float]
    compatibility_score: float
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for your changes
4. Ensure all tests pass
5. Submit a pull request

### Development Setup

```bash
git clone https://github.com/sadiqkhzn/langchain-integration-health.git
cd langchain-integration-health
pip install -e ".[dev]"
pytest tests/
```

## License

MIT License - see LICENSE file for details.

## Support

For issues and feature requests, please use the GitHub issue tracker.
