from typing import List, Dict, Any
import json
import csv
from datetime import datetime
from ..testers.base_tester import IntegrationTestResult

class CompatibilityReporter:
    """Generate compatibility reports in various formats"""
    
    def __init__(self, test_results: List[IntegrationTestResult]):
        self.test_results = test_results
    
    def generate_json_report(self) -> str:
        """Generate JSON compatibility report"""
        report_data = {
            "report_timestamp": datetime.now().isoformat(),
            "total_integrations_tested": len(self.test_results),
            "summary": self._generate_summary(),
            "compatibility_matrix": self._generate_compatibility_matrix(),
            "detailed_results": [
                {
                    "integration_name": result.integration_name,
                    "integration_version": result.integration_version,
                    "test_timestamp": result.test_timestamp.isoformat(),
                    "compatibility_score": result.compatibility_score,
                    "bind_tools_support": result.bind_tools_support,
                    "streaming_support": result.streaming_support,
                    "structured_output_support": result.structured_output_support,
                    "async_support": result.async_support,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "performance_metrics": result.performance_metrics
                }
                for result in self.test_results
            ]
        }
        
        return json.dumps(report_data, indent=2, default=str)
    
    def generate_csv_report(self) -> str:
        """Generate CSV compatibility report"""
        if not self.test_results:
            return "No test results available"
            
        # Define CSV headers
        headers = [
            "integration_name", "integration_version", "test_timestamp",
            "compatibility_score", "bind_tools_support", "streaming_support",
            "structured_output_support", "async_support", "error_count",
            "warning_count", "avg_invoke_latency", "avg_streaming_latency"
        ]
        
        # Prepare CSV data
        csv_data = []
        for result in self.test_results:
            row = [
                result.integration_name,
                result.integration_version,
                result.test_timestamp.isoformat(),
                result.compatibility_score,
                result.bind_tools_support,
                result.streaming_support,
                result.structured_output_support,
                result.async_support,
                len(result.errors),
                len(result.warnings),
                result.performance_metrics.get('invoke_latency', 0),
                result.performance_metrics.get('streaming_latency', 0)
            ]
            csv_data.append(row)
        
        # Generate CSV string
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(csv_data)
        
        return output.getvalue()
    
    def generate_markdown_report(self) -> str:
        """Generate Markdown compatibility report"""
        md_lines = [
            "# LangChain Integration Compatibility Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            ""
        ]
        
        summary = self._generate_summary()
        for key, value in summary.items():
            md_lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        
        md_lines.extend([
            "",
            "## Compatibility Matrix",
            "",
            "| Integration | Version | Score | Bind Tools | Streaming | Structured Output | Async |",
            "|-------------|---------|-------|------------|-----------|-------------------|-------|"
        ])
        
        for result in sorted(self.test_results, key=lambda x: x.compatibility_score, reverse=True):
            score_status = "High" if result.compatibility_score >= 0.8 else "Medium" if result.compatibility_score >= 0.5 else "Low"
            md_lines.append(
                f"| {result.integration_name} | {result.integration_version} | "
                f"{result.compatibility_score:.2f} ({score_status}) | "
                f"{'Yes' if result.bind_tools_support else 'No'} | "
                f"{'Yes' if result.streaming_support else 'No'} | "
                f"{'Yes' if result.structured_output_support else 'No'} | "
                f"{'Yes' if result.async_support else 'No'} |"
            )
        
        # Add detailed results section
        md_lines.extend([
            "",
            "## Detailed Results",
            ""
        ])
        
        for result in self.test_results:
            md_lines.extend([
                f"### {result.integration_name}",
                f"**Version**: {result.integration_version}",
                f"**Compatibility Score**: {result.compatibility_score:.2f}",
                ""
            ])
            
            if result.errors:
                md_lines.extend([
                    "**Errors**:",
                    ""
                ])
                for error in result.errors:
                    md_lines.append(f"- {error}")
                md_lines.append("")
                
            if result.warnings:
                md_lines.extend([
                    "**Warnings**:",
                    ""
                ])
                for warning in result.warnings:
                    md_lines.append(f"- {warning}")
                md_lines.append("")
                
            if result.performance_metrics:
                md_lines.extend([
                    "**Performance Metrics**:",
                    ""
                ])
                for metric, value in result.performance_metrics.items():
                    md_lines.append(f"- {metric}: {value}")
                md_lines.append("")
        
        return "\n".join(md_lines)
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not self.test_results:
            return {"message": "No test results available"}
            
        total = len(self.test_results)
        summary = {
            "total_integrations": total,
            "bind_tools_support": sum(1 for r in self.test_results if r.bind_tools_support),
            "streaming_support": sum(1 for r in self.test_results if r.streaming_support),
            "structured_output_support": sum(1 for r in self.test_results if r.structured_output_support),
            "async_support": sum(1 for r in self.test_results if r.async_support),
            "average_compatibility_score": sum(r.compatibility_score for r in self.test_results) / total,
            "integrations_with_errors": sum(1 for r in self.test_results if r.errors),
            "integrations_with_warnings": sum(1 for r in self.test_results if r.warnings)
        }
        
        return summary
    
    def _generate_compatibility_matrix(self) -> Dict[str, Dict[str, bool]]:
        """Generate compatibility matrix data"""
        matrix = {}
        
        for result in self.test_results:
            matrix[result.integration_name] = {
                "bind_tools": result.bind_tools_support,
                "streaming": result.streaming_support,
                "structured_output": result.structured_output_support,
                "async": result.async_support,
                "compatibility_score": result.compatibility_score
            }
            
        return matrix
    
    def save_report(self, format_type: str, file_path: str) -> None:
        """Save report to file in specified format"""
        if format_type.lower() == "json":
            content = self.generate_json_report()
        elif format_type.lower() == "csv":
            content = self.generate_csv_report()
        elif format_type.lower() == "md" or format_type.lower() == "markdown":
            content = self.generate_markdown_report()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)