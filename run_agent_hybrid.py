import click
import json
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from agent.graph_hybrid import HybridAgent

console = Console()


@click.command()
@click.option('--batch', required=True, type=click.Path(exists=True), help='Input JSONL file with questions')
@click.option('--out', required=True, type=click.Path(), help='Output JSONL file for results')
@click.option('--docs-dir', default='docs', help='Documents directory')
@click.option('--db-path', default='data/northwind.sqlite', help='SQLite database path')
def main(batch, out, docs_dir, db_path):
    console.print(f"\n[bold cyan]Retail Analytics Copilot - Hybrid Agent[/bold cyan]")
    console.print(f"Input: {batch}")
    console.print(f"Output: {out}")
    console.print(f"Docs: {docs_dir}, DB: {db_path}\n")
    console.print("[yellow]Initializing agent...[/yellow]")
    agent = HybridAgent(docs_dir=docs_dir, db_path=db_path)
    console.print("[green]✓ Agent ready[/green]\n")
    questions = []
    with open(batch, 'r') as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    
    console.print(f"Loaded {len(questions)} questions\n")
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Processing questions...", total=len(questions))
        
        for q in questions:
            q_id = q["id"]
            question = q["question"]
            format_hint = q.get("format_hint", "str")
            
            progress.update(task, description=f"[cyan]Processing: {q_id}")
            
            try:
                result = agent.run(question=question, format_hint=format_hint)
                output = {
                    "id": q_id,
                    "final_answer": result["final_answer"],
                    "sql": result["sql"],
                    "confidence": round(result["confidence"], 2),
                    "explanation": result["explanation"][:200],  # Limit to 200 chars
                    "citations": result["citations"]
                }
                
                results.append(output)
                
                console.print(f"[green]✓[/green] {q_id}: {result['final_answer']}")
                
            except Exception as e:
                console.print(f"[red]✗[/red] {q_id}: {e}")
                results.append({
                    "id": q_id,
                    "final_answer": None,
                    "sql": "",
                    "confidence": 0.0,
                    "explanation": f"Error: {str(e)}",
                    "citations": []
                })
            
            progress.advance(task)
    
    with open(out, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    console.print(f"\n[bold green]✓ Complete! Results written to {out}[/bold green]")
    
    success_count = sum(1 for r in results if r["final_answer"] is not None)
    console.print(f"\nSuccess rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")


if __name__ == '__main__':
    main()
