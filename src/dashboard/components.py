import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Any

class CompatibilityMatrix:
    """Component for displaying integration compatibility matrix"""
    
    def __init__(self, test_results: List[Any]):
        self.test_results = test_results
    
    def render(self):
        """Render the compatibility matrix"""
        if not self.test_results:
            st.info("No test results to display")
            return
        
        # Prepare matrix data
        matrix_data = []
        features = ["bind_tools", "streaming", "structured_output", "async"]
        
        for result in self.test_results:
            row = {
                "Integration": result.integration_name,
                "Version": result.integration_version,
                "Score": result.compatibility_score,
                "bind_tools": "Yes" if result.bind_tools_support else "No",
                "streaming": "Yes" if result.streaming_support else "No", 
                "structured_output": "Yes" if result.structured_output_support else "No",
                "async": "Yes" if result.async_support else "No",
                "Errors": len(result.errors),
                "Warnings": len(result.warnings)
            }
            matrix_data.append(row)
        
        df = pd.DataFrame(matrix_data)
        
        # Sort by compatibility score
        df = df.sort_values("Score", ascending=False)
        
        # Color-code the compatibility score
        def score_color(val):
            if val >= 0.8:
                return "background-color: #d4edda"  # Green
            elif val >= 0.5:
                return "background-color: #fff3cd"  # Yellow
            else:
                return "background-color: #f8d7da"  # Red
        
        # Display the matrix
        styled_df = df.style.applymap(score_color, subset=["Score"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_integrations = len(df)
            st.metric("Total Integrations", total_integrations)
        
        with col2:
            high_compatibility = len(df[df["Score"] >= 0.8])
            st.metric("High Compatibility (≥0.8)", f"{high_compatibility}/{total_integrations}")
        
        with col3:
            bind_tools_support = len(df[df["bind_tools"] == "Yes"])
            st.metric("bind_tools Support", f"{bind_tools_support}/{total_integrations}")
        
        with col4:
            streaming_support = len(df[df["streaming"] == "Yes"])
            st.metric("Streaming Support", f"{streaming_support}/{total_integrations}")

class IntegrationDetails:
    """Component for displaying detailed integration information"""
    
    def __init__(self, test_results: List[Any]):
        self.test_results = test_results
    
    def render(self):
        """Render detailed integration results"""
        if not self.test_results:
            st.info("No test results to display")
            return
        
        # Sort by compatibility score
        sorted_results = sorted(self.test_results, key=lambda x: x.compatibility_score, reverse=True)
        
        for result in sorted_results:
            with st.expander(f"{result.integration_name} (Score: {result.compatibility_score:.2f})"):
                
                # Basic information
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Version**: {result.integration_version}")
                    st.write(f"**Test Date**: {result.test_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                
                with col2:
                    st.write(f"**Compatibility Score**: {result.compatibility_score:.2f}")
                    st.write(f"**Errors**: {len(result.errors)}")
                    st.write(f"**Warnings**: {len(result.warnings)}")
                
                with col3:
                    # Feature support indicators
                    features = {
                        "bind_tools": result.bind_tools_support,
                        "streaming": result.streaming_support,
                        "structured_output": result.structured_output_support,
                        "async": result.async_support
                    }
                    
                    for feature, supported in features.items():
                        icon = "Yes" if supported else "No"
                        st.write(f"**{feature}**: {icon}")
                
                # Errors section
                if result.errors:
                    st.subheader("Errors")
                    for error in result.errors:
                        st.error(error)
                
                # Warnings section
                if result.warnings:
                    st.subheader("Warnings")
                    for warning in result.warnings:
                        st.warning(warning)
                
                # Performance metrics
                if result.performance_metrics:
                    st.subheader("Performance Metrics")
                    
                    metrics_df = pd.DataFrame([
                        {"Metric": metric, "Value": value}
                        for metric, value in result.performance_metrics.items()
                    ])
                    
                    st.dataframe(metrics_df, hide_index=True)
                    
                    # Visualize key performance metrics
                    if any(metric in result.performance_metrics for metric in ["invoke_latency", "streaming_latency"]):
                        fig = go.Figure()
                        
                        for metric in ["invoke_latency", "streaming_latency"]:
                            if metric in result.performance_metrics:
                                fig.add_trace(go.Bar(
                                    name=metric.replace('_', ' ').title(),
                                    x=[metric],
                                    y=[result.performance_metrics[metric]]
                                ))
                        
                        fig.update_layout(
                            title="Latency Metrics",
                            yaxis_title="Seconds",
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

class FeatureSupport:
    """Component for visualizing feature support across integrations"""
    
    def __init__(self, test_results: List[Any]):
        self.test_results = test_results
    
    def render(self):
        """Render feature support visualization"""
        if not self.test_results:
            return
        
        # Prepare data for visualization
        features_data = {
            "bind_tools": sum(1 for r in self.test_results if r.bind_tools_support),
            "streaming": sum(1 for r in self.test_results if r.streaming_support),
            "structured_output": sum(1 for r in self.test_results if r.structured_output_support),
            "async": sum(1 for r in self.test_results if r.async_support)
        }
        
        total = len(self.test_results)
        
        # Create bar chart
        fig = px.bar(
            x=list(features_data.keys()),
            y=list(features_data.values()),
            title="Feature Support Across Integrations",
            labels={'x': 'Features', 'y': 'Number of Integrations'}
        )
        
        # Add percentage annotations
        for i, (feature, count) in enumerate(features_data.items()):
            percentage = (count / total) * 100
            fig.add_annotation(
                x=i,
                y=count + 0.1,
                text=f"{percentage:.1f}%",
                showarrow=False
            )
        
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)