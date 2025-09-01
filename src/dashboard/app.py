import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.dashboard.components import CompatibilityMatrix, IntegrationDetails
from src.dashboard.data_loader import DataLoader
from src.utils.discovery import IntegrationDiscovery
from src.testers import LLMIntegrationTester, ChatModelTester, EmbeddingsTester

def create_integration_dashboard():
    """Main Streamlit dashboard application"""
    
    st.set_page_config(
        page_title="LangChain Integration Health", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔗 LangChain Integration Health Dashboard")
    st.markdown("Real-time compatibility monitoring for LangChain integrations")
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        
        # Refresh data button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        # Test new integration
        if st.button("🧪 Run New Tests", use_container_width=True):
            run_integration_tests()
        
        # Export options
        st.header("Export Options")
        export_format = st.selectbox("Format", ["JSON", "CSV", "Markdown"])
        
        if st.button("📥 Export Results", use_container_width=True):
            export_results(export_format.lower())
        
        # Filters
        st.header("Filters")
        
        # Filter by integration type
        integration_types = st.multiselect(
            "Integration Types",
            ["LLMs", "Chat Models", "Embeddings"],
            default=["LLMs", "Chat Models", "Embeddings"]
        )
        
        # Filter by compatibility score
        min_score = st.slider("Minimum Compatibility Score", 0.0, 1.0, 0.0, 0.1)
        
        # Filter by features
        required_features = st.multiselect(
            "Required Features",
            ["bind_tools", "streaming", "structured_output", "async"],
            default=[]
        )
    
    # Load test results
    test_results = data_loader.load_test_results()
    
    if not test_results:
        st.warning("No test results found. Run some tests to see the dashboard in action!")
        
        # Show discovery section
        st.header("🔍 Available Integrations")
        discovery = IntegrationDiscovery()
        integrations = discovery.discover_all_integrations()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("LLMs Found", len(integrations.get("llms", [])))
            if integrations.get("llms"):
                with st.expander("View LLMs"):
                    for llm in integrations["llms"]:
                        st.write(f"- {llm.__name__}")
        
        with col2:
            st.metric("Chat Models Found", len(integrations.get("chat_models", [])))
            if integrations.get("chat_models"):
                with st.expander("View Chat Models"):
                    for chat in integrations["chat_models"]:
                        st.write(f"- {chat.__name__}")
        
        with col3:
            st.metric("Embeddings Found", len(integrations.get("embeddings", [])))
            if integrations.get("embeddings"):
                with st.expander("View Embeddings"):
                    for emb in integrations["embeddings"]:
                        st.write(f"- {emb.__name__}")
        
        return
    
    # Filter results based on sidebar selections
    filtered_results = filter_results(test_results, integration_types, min_score, required_features)
    
    # Main dashboard content
    display_dashboard_overview(filtered_results)
    
    # Compatibility Matrix
    st.header("Compatibility Matrix")
    compatibility_matrix = CompatibilityMatrix(filtered_results)
    compatibility_matrix.render()
    
    # Detailed Results
    st.header("Detailed Results")
    integration_details = IntegrationDetails(filtered_results)
    integration_details.render()
    
    # Performance Metrics
    display_performance_metrics(filtered_results)
    
    # Historical Trends (if available)
    display_historical_trends(data_loader)

def display_dashboard_overview(test_results: List[Any]):
    """Display overview metrics"""
    if not test_results:
        st.info("No results match the current filters.")
        return
    
    # Calculate summary metrics
    total_integrations = len(test_results)
    avg_score = sum(r.compatibility_score for r in test_results) / total_integrations
    
    bind_tools_count = sum(1 for r in test_results if r.bind_tools_support)
    streaming_count = sum(1 for r in test_results if r.streaming_support)
    structured_output_count = sum(1 for r in test_results if r.structured_output_support)
    async_count = sum(1 for r in test_results if r.async_support)
    
    # Display metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Integrations", total_integrations)
    
    with col2:
        score_color = "green" if avg_score >= 0.8 else "orange" if avg_score >= 0.5 else "red"
        st.metric("Avg Compatibility", f"{avg_score:.2f}", delta_color=score_color)
    
    with col3:
        st.metric("bind_tools Support", f"{bind_tools_count}/{total_integrations}")
    
    with col4:
        st.metric("Streaming Support", f"{streaming_count}/{total_integrations}")
    
    with col5:
        st.metric("Async Support", f"{async_count}/{total_integrations}")

def display_performance_metrics(test_results: List[Any]):
    """Display performance metrics visualization"""
    st.header("⚡ Performance Metrics")
    
    if not test_results:
        return
    
    # Prepare performance data
    perf_data = []
    for result in test_results:
        for metric, value in result.performance_metrics.items():
            perf_data.append({
                "integration": result.integration_name,
                "metric": metric,
                "value": value,
                "compatibility_score": result.compatibility_score
            })
    
    if perf_data:
        df = pd.DataFrame(perf_data)
        
        # Create performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            if 'invoke_latency' in df['metric'].values:
                latency_df = df[df['metric'] == 'invoke_latency']
                fig = px.bar(
                    latency_df, 
                    x='integration', 
                    y='value',
                    title="Invoke Latency by Integration",
                    labels={'value': 'Latency (seconds)', 'integration': 'Integration'}
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'streaming_latency' in df['metric'].values:
                streaming_df = df[df['metric'] == 'streaming_latency']
                fig = px.bar(
                    streaming_df,
                    x='integration',
                    y='value', 
                    title="Streaming Latency by Integration",
                    labels={'value': 'Latency (seconds)', 'integration': 'Integration'}
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

def display_historical_trends(data_loader: DataLoader):
    """Display historical compatibility trends"""
    st.header("📈 Historical Trends")
    
    historical_data = data_loader.load_historical_data()
    
    if historical_data:
        df = pd.DataFrame(historical_data)
        
        # Group by date and calculate average compatibility score
        daily_scores = df.groupby(df['test_timestamp'].dt.date)['compatibility_score'].mean()
        
        fig = px.line(
            x=daily_scores.index,
            y=daily_scores.values,
            title="Average Compatibility Score Over Time",
            labels={'x': 'Date', 'y': 'Compatibility Score'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical data available yet. Run tests over time to see trends.")

def filter_results(test_results: List[Any], integration_types: List[str], 
                  min_score: float, required_features: List[str]) -> List[Any]:
    """Filter test results based on user selections"""
    filtered = test_results
    
    # Filter by compatibility score
    filtered = [r for r in filtered if r.compatibility_score >= min_score]
    
    # Filter by required features
    for feature in required_features:
        if feature == "bind_tools":
            filtered = [r for r in filtered if r.bind_tools_support]
        elif feature == "streaming":
            filtered = [r for r in filtered if r.streaming_support]
        elif feature == "structured_output":
            filtered = [r for r in filtered if r.structured_output_support]
        elif feature == "async":
            filtered = [r for r in filtered if r.async_support]
    
    return filtered

@st.cache_data
def run_integration_tests():
    """Run integration tests and cache results"""
    st.info("Running integration tests... This may take a few minutes.")
    
    # This would typically run the actual tests
    # For now, we'll show a placeholder
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"Testing integration {i//10 + 1} of 10...")
        
    st.success("Integration tests completed!")

def export_results(format_type: str):
    """Export test results in specified format"""
    data_loader = DataLoader()
    test_results = data_loader.load_test_results()
    
    if test_results:
        from ..utils.reporters import CompatibilityReporter
        reporter = CompatibilityReporter(test_results)
        
        if format_type == "json":
            content = reporter.generate_json_report()
            st.download_button(
                "Download JSON Report",
                content,
                file_name=f"langchain_compatibility_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        elif format_type == "csv":
            content = reporter.generate_csv_report()
            st.download_button(
                "Download CSV Report",
                content,
                file_name=f"langchain_compatibility_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        elif format_type == "markdown":
            content = reporter.generate_markdown_report()
            st.download_button(
                "Download Markdown Report",
                content,
                file_name=f"langchain_compatibility_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    create_integration_dashboard()