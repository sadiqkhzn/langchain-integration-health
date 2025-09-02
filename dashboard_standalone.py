#!/usr/bin/env python3
"""
Standalone Streamlit dashboard for LangChain Integration Health
Run with: streamlit run dashboard_standalone.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

def main():
    st.set_page_config(
        page_title="LangChain Integration Health", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("LangChain Integration Health Dashboard")
    st.markdown("Real-time compatibility monitoring for LangChain integrations")
    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        
        if st.button("Refresh Data", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("Run New Tests", width="stretch"):
            st.info("Integration testing would run here!")
        
        st.header("Export Options")
        export_format = st.selectbox("Format", ["JSON", "CSV", "Markdown"])
        
        if st.button("Export Results", width="stretch"):
            st.info(f"Would export in {export_format} format")
        
        st.header("Filters")
        integration_types = st.multiselect(
            "Integration Types",
            ["LLMs", "Chat Models", "Embeddings"],
            default=["LLMs", "Chat Models", "Embeddings"]
        )
        
        min_score = st.slider("Minimum Compatibility Score", 0.0, 1.0, 0.0, 0.1)
    
    # Demo data for the dashboard
    demo_data = create_demo_data()
    
    if not demo_data:
        st.warning("No test results found. Run some tests to see the dashboard in action!")
        
        # Show discovery info
        st.header("Framework Capabilities")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Supported Integrations", "50+")
            st.info("LLMs, Chat Models, Embeddings")
        
        with col2:
            st.metric("Test Coverage", "6 Core Methods")
            st.info("bind_tools, streaming, structured output, async, invoke, error handling")
        
        with col3:
            st.metric("Export Formats", "3")
            st.info("JSON, CSV, Markdown")
        
        # Show MLXPipeline fix example
        st.header("MLXPipeline Fix Example")
        st.code('''
from src.examples.mlx_pipeline_fix import create_mlx_wrapper

# Original MLXPipeline (missing bind_tools)
mlx = MLXPipeline(model_name="mlx-community/Llama-3.2-1B-Instruct-4bit")

# Wrap for LangChain compatibility  
langchain_mlx = create_mlx_wrapper(mlx)

# Now you can use bind_tools!
mlx_with_tools = langchain_mlx.bind_tools([calculator_tool])
response = mlx_with_tools.invoke("What is 5 + 3?")
        ''', language="python")
        
        return
    
    # Filter demo data
    filtered_data = filter_demo_data(demo_data, min_score)
    
    # Overview metrics
    display_overview(filtered_data)
    
    # Compatibility Matrix
    st.header("Compatibility Matrix")
    display_compatibility_matrix(filtered_data)
    
    # Detailed Results
    st.header("Detailed Results")
    display_detailed_results(filtered_data)
    
    # Performance Metrics
    st.header("Performance Metrics")
    display_performance_metrics(filtered_data)

def create_demo_data():
    """Create demo test results for display"""
    return [
        {
            "integration_name": "ChatOpenAI",
            "integration_version": "0.1.0",
            "compatibility_score": 0.95,
            "bind_tools_support": True,
            "streaming_support": True,
            "structured_output_support": True,
            "async_support": True,
            "errors": [],
            "warnings": [],
            "performance_metrics": {
                "invoke_latency": 0.8,
                "streaming_latency": 1.2,
                "chunks_received": 15
            }
        },
        {
            "integration_name": "Anthropic",
            "integration_version": "0.2.1",
            "compatibility_score": 0.90,
            "bind_tools_support": True,
            "streaming_support": True,
            "structured_output_support": True,
            "async_support": True,
            "errors": [],
            "warnings": ["Minor deprecation warning"],
            "performance_metrics": {
                "invoke_latency": 1.1,
                "streaming_latency": 1.5
            }
        },
        {
            "integration_name": "MLXPipeline",
            "integration_version": "0.1.0",
            "compatibility_score": 0.35,
            "bind_tools_support": False,
            "streaming_support": True,
            "structured_output_support": False,
            "async_support": False,
            "errors": ["Missing bind_tools method", "No structured output support"],
            "warnings": ["Limited async support"],
            "performance_metrics": {
                "invoke_latency": 2.1
            }
        },
        {
            "integration_name": "HuggingFacePipeline",
            "integration_version": "1.0.0",
            "compatibility_score": 0.75,
            "bind_tools_support": True,
            "streaming_support": False,
            "structured_output_support": True,
            "async_support": True,
            "errors": [],
            "warnings": ["Streaming not fully implemented"],
            "performance_metrics": {
                "invoke_latency": 3.2
            }
        },
        {
            "integration_name": "OpenAIEmbeddings",
            "integration_version": "0.1.5",
            "compatibility_score": 0.85,
            "bind_tools_support": False,
            "streaming_support": False,
            "structured_output_support": False,
            "async_support": True,
            "errors": [],
            "warnings": ["Embeddings don't support tools/streaming"],
            "performance_metrics": {
                "embedding_latency": 0.5,
                "documents_embedded": 100
            }
        }
    ]

def filter_demo_data(data, min_score):
    """Filter demo data based on score"""
    return [item for item in data if item["compatibility_score"] >= min_score]

def display_overview(data):
    """Display overview metrics"""
    total = len(data)
    avg_score = sum(item["compatibility_score"] for item in data) / total if total > 0 else 0
    bind_tools_count = sum(1 for item in data if item["bind_tools_support"])
    streaming_count = sum(1 for item in data if item["streaming_support"])
    async_count = sum(1 for item in data if item["async_support"])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Integrations", total)
    
    with col2:
        st.metric("Avg Compatibility", f"{avg_score:.2f}")
    
    with col3:
        st.metric("bind_tools Support", f"{bind_tools_count}/{total}")
    
    with col4:
        st.metric("Streaming Support", f"{streaming_count}/{total}")
    
    with col5:
        st.metric("Async Support", f"{async_count}/{total}")

def display_compatibility_matrix(data):
    """Display compatibility matrix"""
    if not data:
        st.info("No data to display")
        return
    
    # Create DataFrame
    matrix_data = []
    for item in data:
        matrix_data.append({
            "Integration": item["integration_name"],
            "Version": item["integration_version"],
            "Score": item["compatibility_score"],
            "bind_tools": "Yes" if item["bind_tools_support"] else "No",
            "streaming": "Yes" if item["streaming_support"] else "No",
            "structured_output": "Yes" if item["structured_output_support"] else "No",
            "async": "Yes" if item["async_support"] else "No",
            "Errors": len(item["errors"]),
            "Warnings": len(item["warnings"])
        })
    
    df = pd.DataFrame(matrix_data)
    df = df.sort_values("Score", ascending=False)
    
    # Style the dataframe
    def score_color(val):
        if val >= 0.8:
            return "background-color: #d4edda"
        elif val >= 0.5:
            return "background-color: #fff3cd"
        else:
            return "background-color: #f8d7da"
    
    styled_df = df.style.map(score_color, subset=["Score"])
    st.dataframe(styled_df, width="stretch", hide_index=True)

def display_detailed_results(data):
    """Display detailed results with expandable sections"""
    for item in sorted(data, key=lambda x: x["compatibility_score"], reverse=True):
        with st.expander(f"{item['integration_name']} (Score: {item['compatibility_score']:.2f})"):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Version**: {item['integration_version']}")
                st.write(f"**Compatibility Score**: {item['compatibility_score']:.2f}")
            
            with col2:
                st.write(f"**Errors**: {len(item['errors'])}")
                st.write(f"**Warnings**: {len(item['warnings'])}")
            
            with col3:
                features = {
                    "bind_tools": item["bind_tools_support"],
                    "streaming": item["streaming_support"], 
                    "structured_output": item["structured_output_support"],
                    "async": item["async_support"]
                }
                
                for feature, supported in features.items():
                    status = "Yes" if supported else "No"
                    st.write(f"**{feature}**: {status}")
            
            # Show errors
            if item["errors"]:
                st.subheader("Errors")
                for error in item["errors"]:
                    st.error(error)
            
            # Show warnings
            if item["warnings"]:
                st.subheader("Warnings")
                for warning in item["warnings"]:
                    st.warning(warning)
            
            # Show performance metrics
            if item["performance_metrics"]:
                st.subheader("Performance Metrics")
                metrics_df = pd.DataFrame([
                    {"Metric": metric, "Value": value}
                    for metric, value in item["performance_metrics"].items()
                ])
                st.dataframe(metrics_df, hide_index=True)

def display_performance_metrics(data):
    """Display performance visualizations"""
    if not data:
        return
    
    # Extract latency data
    latency_data = []
    for item in data:
        if "invoke_latency" in item["performance_metrics"]:
            latency_data.append({
                "Integration": item["integration_name"],
                "Invoke Latency": item["performance_metrics"]["invoke_latency"]
            })
    
    if latency_data:
        df = pd.DataFrame(latency_data)
        fig = px.bar(
            df,
            x="Integration",
            y="Invoke Latency", 
            title="Invoke Latency by Integration",
            labels={'Invoke Latency': 'Latency (seconds)'}
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, width='stretch')

if __name__ == "__main__":
    main()