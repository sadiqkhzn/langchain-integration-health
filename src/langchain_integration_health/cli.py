#!/usr/bin/env python3
"""
Command-line interface for LangChain Integration Health Dashboard & Testing Framework
"""

import asyncio
import typer
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path

from .utils.config import Config
from .utils.discovery import IntegrationDiscovery
from .utils.reporters import CompatibilityReporter
from .testers import LLMIntegrationTester, ChatModelTester, EmbeddingsTester
from .dashboard.data_loader import DataLoader

app = typer.Typer(help="LangChain Integration Health Dashboard & Testing Framework")
console = Console()

@app.command()
def discover(
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for results"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv")
):
    """Discover available LangChain integrations"""
    
    console.print("Discovering LangChain integrations...")
    
    discovery = IntegrationDiscovery()
    integrations = discovery.discover_all_integrations()
    
    total_found = sum(len(classes) for classes in integrations.values())
    console.print(f"Found {total_found} integrations")
    
    if format == "table":
        for category, classes in integrations.items():
            if classes:
                table = Table(title=f"{category.upper()} Integrations")
                table.add_column("Name", style="cyan")
                table.add_column("Module", style="magenta")
                
                for cls in classes:
                    table.add_row(cls.__name__, cls.__module__)
                
                console.print(table)
    
    elif format == "json":
        import json
        output_data = {
            category: [{"name": cls.__name__, "module": cls.__module__} for cls in classes]
            for category, classes in integrations.items()
        }
        
        if output:
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
            console.print(f"Results saved to {output}")
        else:
            console.print(json.dumps(output_data, indent=2))

@app.command()
def test(
    integration: Optional[str] = typer.Option(None, "--integration", "-i", help="Specific integration to test"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Integration category: llms, chat_models, embeddings"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for results"),
    mock: bool = typer.Option(False, "--mock", help="Run in mock mode without API calls"),
    parallel: bool = typer.Option(True, "--parallel", help="Run tests in parallel")
):
    """Run integration compatibility tests"""
    
    config = Config.from_env()
    config.mock_mode = mock
    config.parallel_tests = parallel
    
    discovery = IntegrationDiscovery()
    integrations = discovery.discover_all_integrations()
    
    # Filter integrations based on parameters
    integrations_to_test = []
    
    for cat, classes in integrations.items():
        if category and cat != category:
            continue
            
        for cls in classes:
            if integration and cls.__name__ != integration:
                continue
            integrations_to_test.append((cat, cls))
    
    if not integrations_to_test:
        console.print("No integrations found matching criteria")
        return
    
    console.print(f"Testing {len(integrations_to_test)} integrations...")
    
    async def run_tests():
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            tasks = []
            for category, integration_class in integrations_to_test:
                task = progress.add_task(f"Testing {integration_class.__name__}...")
                
                # Select appropriate tester
                if category == "llms":
                    tester = LLMIntegrationTester(integration_class, config.get_integration_config(integration_class.__name__))
                elif category == "chat_models":
                    tester = ChatModelTester(integration_class, config.get_integration_config(integration_class.__name__))
                elif category == "embeddings":
                    tester = EmbeddingsTester(integration_class, config.get_integration_config(integration_class.__name__))
                else:
                    continue
                
                # Run test
                if parallel:
                    tasks.append(asyncio.create_task(tester.run_all_tests()))
                else:
                    result = await tester.run_all_tests()
                    results.append(result)
                    progress.update(task, completed=True)
            
            # Wait for parallel tests to complete
            if parallel and tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Filter out exceptions
                results = [r for r in results if isinstance(r, IntegrationTestResult)]
        
        return results
    
    # Run tests
    results = asyncio.run(run_tests())
    
    # Display results
    if results:
        # Save to database
        data_loader = DataLoader()
        for result in results:
            data_loader.save_test_result(result)
        
        # Generate report
        reporter = CompatibilityReporter(results)
        
        if output:
            if output.endswith('.json'):
                reporter.save_report('json', output)
            elif output.endswith('.csv'):
                reporter.save_report('csv', output)
            elif output.endswith('.md'):
                reporter.save_report('markdown', output)
            else:
                reporter.save_report('json', output)
            
            console.print(f"Results saved to {output}")
        else:
            # Display summary table
            table = Table(title="Integration Test Results")
            table.add_column("Integration", style="cyan")
            table.add_column("Score", style="bold")
            table.add_column("bind_tools", justify="center")
            table.add_column("Streaming", justify="center")
            table.add_column("Structured Output", justify="center")
            table.add_column("Async", justify="center")
            table.add_column("Errors", style="red")
            
            for result in sorted(results, key=lambda x: x.compatibility_score, reverse=True):
                score_style = "green" if result.compatibility_score >= 0.8 else "yellow" if result.compatibility_score >= 0.5 else "red"
                
                table.add_row(
                    result.integration_name,
                    f"{result.compatibility_score:.2f}",
                    "Yes" if result.bind_tools_support else "No",
                    "Yes" if result.streaming_support else "No",
                    "Yes" if result.structured_output_support else "No",
                    "Yes" if result.async_support else "No",
                    str(len(result.errors))
                )
            
            console.print(table)

@app.command()
def dashboard(
    host: str = typer.Option("localhost", "--host", help="Dashboard host"),
    port: int = typer.Option(8501, "--port", help="Dashboard port")
):
    """Launch the Streamlit dashboard"""
    
    import subprocess
    import sys
    
    console.print(f"Launching dashboard at http://{host}:{port}")
    
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.address", host,
        "--server.port", str(port)
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        console.print("\nDashboard stopped")

@app.command()
def report(
    format: str = typer.Option("markdown", "--format", "-f", help="Report format: json, csv, markdown"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
    days: int = typer.Option(7, "--days", "-d", help="Include results from last N days")
):
    """Generate compatibility report"""
    
    data_loader = DataLoader()
    
    # Load recent results
    all_results = data_loader.load_test_results()
    
    if not all_results:
        console.print("No test results found. Run tests first with: langchain-health test")
        return
    
    # Filter by days
    from datetime import datetime, timedelta
    cutoff = datetime.now() - timedelta(days=days)
    recent_results = [r for r in all_results if r.test_timestamp >= cutoff]
    
    if not recent_results:
        console.print(f"No test results found in the last {days} days")
        return
    
    console.print(f"Generating {format} report for {len(recent_results)} results...")
    
    reporter = CompatibilityReporter(recent_results)
    
    if output:
        reporter.save_report(format, output)
        console.print(f"Report saved to {output}")
    else:
        if format == "json":
            console.print(reporter.generate_json_report())
        elif format == "csv":
            console.print(reporter.generate_csv_report())
        elif format == "markdown":
            console.print(reporter.generate_markdown_report())

@app.command()
def clean(
    days: int = typer.Option(90, "--days", "-d", help="Delete results older than N days"),
    confirm: bool = typer.Option(False, "--confirm", "-y", help="Skip confirmation prompt")
):
    """Clean old test results from database"""
    
    if not confirm:
        if not typer.confirm(f"Delete test results older than {days} days?"):
            console.print("Operation cancelled")
            return
    
    data_loader = DataLoader()
    deleted_count = data_loader.delete_old_results(days)
    
    console.print(f"Deleted {deleted_count} old test results")

def main():
    """Main entry point"""
    app()

if __name__ == "__main__":
    main()