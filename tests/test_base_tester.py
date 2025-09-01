import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime
from src.testers.base_tester import BaseIntegrationTester, IntegrationTestResult

class MockIntegrationTester(BaseIntegrationTester):
    """Mock tester for testing base functionality"""
    
    REQUIRED_METHODS = ["invoke", "stream"]
    
    async def _test_method_functionality(self):
        return {"invoke": True, "stream": True}
    
    async def _test_specific_error_scenarios(self, instance):
        return {"handles_errors": True}

class MockIntegration:
    """Mock integration class for testing"""
    
    def __init__(self, **kwargs):
        pass
    
    def invoke(self, text):
        return f"Response to: {text}"
    
    def stream(self, text):
        for word in text.split():
            yield word

@pytest.fixture
def mock_tester():
    return MockIntegrationTester(MockIntegration)

@pytest.mark.asyncio
async def test_base_tester_initialization(mock_tester):
    """Test that base tester initializes correctly"""
    assert mock_tester.integration_class == MockIntegration
    assert mock_tester.config == {}
    assert isinstance(mock_tester.results, IntegrationTestResult)
    assert mock_tester.results.integration_name == "MockIntegration"

@pytest.mark.asyncio
async def test_run_all_tests(mock_tester):
    """Test that run_all_tests executes all test phases"""
    result = await mock_tester.run_all_tests()
    
    assert isinstance(result, IntegrationTestResult)
    assert result.integration_name == "MockIntegration"
    assert isinstance(result.test_timestamp, datetime)
    assert isinstance(result.compatibility_score, float)

@pytest.mark.asyncio
async def test_instantiation_test(mock_tester):
    """Test integration instantiation testing"""
    success = await mock_tester._test_instantiation()
    assert success is True

def test_required_methods_test(mock_tester):
    """Test required methods validation"""
    results = mock_tester._test_required_methods()
    
    assert "invoke" in results
    assert "stream" in results
    assert results["invoke"] is True
    assert results["stream"] is True

@pytest.mark.asyncio
async def test_compatibility_score_calculation(mock_tester):
    """Test compatibility score calculation"""
    # Set some test values
    mock_tester.results.bind_tools_support = True
    mock_tester.results.streaming_support = True
    mock_tester.results.structured_output_support = False
    mock_tester.results.async_support = False
    
    score = mock_tester._calculate_compatibility_score()
    
    # Should be 0.7 * (2/4) = 0.35
    assert score == pytest.approx(0.35, rel=1e-2)
    assert mock_tester.results.compatibility_score == score

def test_async_support_check(mock_tester):
    """Test async support detection"""
    # Mock integration doesn't have async methods
    has_async_invoke = mock_tester._check_async_support("ainvoke")
    assert has_async_invoke is False
    
    # Test with existing sync method
    has_sync_invoke = mock_tester._check_async_support("invoke")
    assert has_sync_invoke is False  # invoke is not async

class MockAsyncIntegration:
    """Mock integration with async methods"""
    
    def __init__(self, **kwargs):
        pass
    
    async def ainvoke(self, text):
        return f"Async response to: {text}"

def test_async_support_detection():
    """Test detection of async methods"""
    tester = MockIntegrationTester(MockAsyncIntegration)
    has_async = tester._check_async_support("ainvoke")
    assert has_async is True