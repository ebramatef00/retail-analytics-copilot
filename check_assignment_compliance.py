#!/usr/bin/env python3
"""
Check if the project complies with assignment requirements
Specifically: Are we using DSPy properly or relying on hardcoded logic?
"""

import re
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def check_file_for_hardcoding(filepath: str) -> dict:
    """Check a file for hardcoded logic patterns"""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    issues = []
    
    # Check for hardcoded routing
    if 'if "according to" in' in content and 'route =' in content:
        issues.append("❌ Hardcoded routing logic detected")
    
    # Check for hardcoded SQL templates
    sql_templates = re.findall(r'if.*in question_lower.*:\s*sql = ["\']SELECT', content, re.MULTILINE)
    if sql_templates:
        issues.append(f"❌ Found {len(sql_templates)} hardcoded SQL templates")
    
    # Check for proper DSPy usage
    if 'self.router(' in content or 'self.sql_generator(' in content:
        issues.append("✓ Using DSPy modules")
    
    # Check for DSPy optimization
    if 'BootstrapFewShot' in content or 'teleprompter' in content:
        issues.append("✓ DSPy optimization present")
    
    return {
        'path': filepath,
        'issues': issues,
        'content_lines': len(content.split('\n'))
    }


def main():
    console.print(Panel.fit(
        "[bold cyan]Assignment Compliance Checker[/bold cyan]\n"
        "Verifying DSPy usage vs hardcoded logic",
        border_style="cyan"
    ))
    
    files_to_check = [
        'agent/graph_hybrid.py',
        'agent/dspy_signatures.py',
        'optimize_agent.py',
    ]
    
    results = []
    
    console.print("\n[yellow]Checking files...[/yellow]\n")
    
    for filepath in files_to_check:
        if not os.path.exists(filepath):
            console.print(f"[red]✗ Missing: {filepath}[/red]")
            continue
        
        result = check_file_for_hardcoding(filepath)
        results.append(result)
        
        console.print(f"[cyan]{filepath}[/cyan] ({result['content_lines']} lines)")
        for issue in result['issues']:
            if issue.startswith('✓'):
                console.print(f"  [green]{issue}[/green]")
            else:
                console.print(f"  [red]{issue}[/red]")
        console.print()
    
    # Summary
    console.print("="*60)
    console.print("[bold]Compliance Summary[/bold]")
    console.print("="*60)
    
    all_issues = [issue for r in results for issue in r['issues']]
    
    has_dspy = any('Using DSPy modules' in i for i in all_issues)
    has_optimization = any('optimization present' in i for i in all_issues)
    has_hardcoding = any('Hardcoded' in i for i in all_issues)
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Requirement", style="cyan")
    table.add_column("Status", justify="center")
    
    table.add_row(
        "Using DSPy modules",
        "[green]✓[/green]" if has_dspy else "[red]✗[/red]"
    )
    table.add_row(
        "DSPy optimization script",
        "[green]✓[/green]" if has_optimization else "[red]✗[/red]"
    )
    table.add_row(
        "No hardcoded logic",
        "[green]✓[/green]" if not has_hardcoding else "[yellow]⚠ Found hardcoding[/yellow]"
    )
    
    console.print(table)
    
    if has_hardcoding:
        console.print("\n[bold yellow]⚠ Warning: Hardcoded logic detected![/bold yellow]")
        console.print("The assignment requires using DSPy's learning capabilities.")
        console.print("Consider replacing hardcoded if/else logic with DSPy modules.")
    else:
        console.print("\n[bold green]✓ Project appears to follow assignment requirements![/bold green]")
    
    # Check for required files
    console.print("\n[cyan]Required Files:[/cyan]")
    required_files = [
        'agent/graph_hybrid.py',
        'agent/dspy_signatures.py',
        'agent/rag/retrieval.py',
        'agent/tools/sqlite_tool.py',
        'data/northwind.sqlite',
        'docs/marketing_calendar.md',
        'docs/kpi_definitions.md',
        'docs/catalog.md',
        'docs/product_policy.md',
        'sample_questions_hybrid_eval.jsonl',
        'run_agent_hybrid.py',
        'README.md',
    ]
    
    missing = []
    for f in required_files:
        if os.path.exists(f):
            console.print(f"  [green]✓[/green] {f}")
        else:
            console.print(f"  [red]✗[/red] {f}")
            missing.append(f)
    
    if missing:
        console.print(f"\n[red]Missing {len(missing)} required files[/red]")
    else:
        console.print(f"\n[green]✓ All required files present[/green]")


if __name__ == "__main__":
    main()