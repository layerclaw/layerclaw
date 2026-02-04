"""CLI tool for Tracer - Deep Training Observability."""

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from tracer import __version__
from tracer.analysis.anomaly_detection import AnomalyDetector
from tracer.analysis.query import QueryEngine

console = Console()


@click.group()
@click.option("--storage", default="./.tracer", help="Storage path")
@click.pass_context
def cli(ctx: Any, storage: str) -> None:
    """
    Tracer CLI - Deep Training Observability Tool.

    Track, debug, and optimize your PyTorch training runs.

    Use 'tracer COMMAND --help' for more information on a command.
    """
    ctx.ensure_object(dict)
    ctx.obj["storage"] = storage
    ctx.obj["query"] = QueryEngine(storage)


@cli.command()
@click.option("--project", help="Filter by project name")
@click.option("--status", help="Filter by status")
@click.option("--limit", default=20, help="Max runs to show")
@click.pass_context
def list(ctx: Any, project: Optional[str], status: Optional[str], limit: int) -> None:
    """List all training runs."""
    query = ctx.obj["query"]

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        task = progress.add_task("Loading runs...", total=None)
        runs = query.storage.list_runs(project_name=project, status=status, limit=limit)
        progress.update(task, completed=True)

    if not runs:
        console.print("[yellow]No runs found[/yellow]")
        return

    # Create table
    table = Table(
        title=f"Training Runs ({len(runs)} found)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Run Name", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Steps", justify="right", style="green")
    table.add_column("Duration", justify="right")
    table.add_column("Started", style="dim")

    for run in runs:
        # Status with emoji
        status_map = {
            "completed": ("✓", "green"),
            "running": ("▶", "yellow"),
            "failed": ("✗", "red"),
            "crashed": ("💥", "red bold"),
        }
        status_emoji, status_style = status_map.get(run["status"], ("?", "white"))
        status_str = f"[{status_style}]{status_emoji} {run['status']}[/{status_style}]"

        # Duration
        if run["end_time"]:
            duration = run["end_time"] - run["start_time"]
            if duration < 60:
                duration_str = f"{duration:.0f}s"
            elif duration < 3600:
                duration_str = f"{duration/60:.1f}m"
            else:
                duration_str = f"{duration/3600:.1f}h"
        else:
            duration_str = "[dim]running[/dim]"

        # Start time
        start_dt = datetime.datetime.fromtimestamp(run["start_time"])
        start_str = start_dt.strftime("%Y-%m-%d %H:%M")

        table.add_row(run["run_name"], status_str, str(run["total_steps"]), duration_str, start_str)

    console.print(table)


@cli.command()
@click.argument("run_name")
@click.option("--metrics", "-m", multiple=True, help="Show specific metrics")
@click.pass_context
def show(ctx: Any, run_name: str, metrics: tuple) -> None:
    """Show detailed information about a run."""
    query = ctx.obj["query"]

    # Find run
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        task = progress.add_task("Loading run...", total=None)
        runs = query.storage.list_runs()
        run = next((r for r in runs if r["run_name"] == run_name), None)
        progress.update(task, completed=True)

    if run is None:
        console.print(f"[red]✗ Run '{run_name}' not found[/red]")
        console.print("\n[dim]Use 'tracer list' to see available runs[/dim]")
        return

    summary = query.get_run_summary(run["run_id"])

    # Header
    console.print(
        Panel(
            f"[bold cyan]{summary['run_name']}[/bold cyan]\n[dim]{summary['run_id']}[/dim]",
            title="Run Details",
            border_style="cyan",
        )
    )

    # Status info
    status_map = {
        "completed": "✓ Completed",
        "running": "▶ Running",
        "failed": "✗ Failed",
        "crashed": "💥 Crashed",
    }
    status_str = status_map.get(summary["status"], summary["status"])

    console.print(f"\n[bold]Status:[/bold] {status_str}")
    console.print(f"[bold]Project:[/bold] {summary['project_name']}")
    console.print(f"[bold]Steps:[/bold] {summary['total_steps']:,}")
    console.print(f"[bold]Checkpoints:[/bold] {summary['total_checkpoints']}")
    console.print(f"[bold]Duration:[/bold] {summary['duration_seconds']:.2f}s")

    # Start/End time
    start_dt = datetime.datetime.fromtimestamp(summary["start_time"])
    console.print(f"[bold]Started:[/bold] {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    if summary["end_time"]:
        end_dt = datetime.datetime.fromtimestamp(summary["end_time"])
        console.print(f"[bold]Ended:[/bold] {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    if summary["config"]:
        console.print("\n[bold]Configuration:[/bold]")
        config_table = Table(show_header=False, box=None, padding=(0, 2))
        config_table.add_column(style="cyan")
        config_table.add_column()

        important_keys = ["checkpoint_interval", "track_gradients", "track_system_metrics"]
        for key in important_keys:
            if key in summary["config"]:
                config_table.add_row(key, str(summary["config"][key]))

        console.print(config_table)

    # Final metrics
    if summary["final_metrics"]:
        console.print("\n[bold]Final Metrics:[/bold]")
        metrics_table = Table(show_header=False, box=None, padding=(0, 2))
        metrics_table.add_column(style="green")
        metrics_table.add_column(justify="right")

        for key, value in summary["final_metrics"].items():
            value_str = f"{value:.6f}" if isinstance(value, float) else str(value)
            metrics_table.add_row(key, value_str)

        console.print(metrics_table)

    # Anomalies
    if summary["anomalies"]:
        console.print("\n[bold yellow]⚠ Anomalies Detected:[/bold yellow]")
        for anomaly_type, count in summary["anomalies"].items():
            console.print(f"  {anomaly_type.replace('_', ' ').title()}: {count}")

    # Show specific metrics if requested
    if metrics:
        console.print("\n[bold]Metric Time Series:[/bold]")
        for metric_name in metrics:
            ts = query.storage.get_metric_timeseries(run["run_id"], metric_name)
            if ts["values"]:
                console.print(f"\n[cyan]{metric_name}:[/cyan]")
                console.print(f"  Initial: {ts['values'][0]:.6f}")
                console.print(f"  Final: {ts['values'][-1]:.6f}")
                console.print(f"  Min: {min(ts['values']):.6f}")
                console.print(f"  Max: {max(ts['values']):.6f}")
            else:
                console.print(f"\n[yellow]No data for metric '{metric_name}'[/yellow]")


@cli.command()
@click.argument("run1")
@click.argument("run2")
@click.option("--metric", "-m", default="loss", help="Metric to compare")
@click.pass_context
def compare(ctx: Any, run1: str, run2: str, metric: str) -> None:
    """Compare two training runs."""
    query = ctx.obj["query"]

    # Find runs
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        task = progress.add_task("Loading runs...", total=None)
        runs = query.storage.list_runs()
        run1_obj = next((r for r in runs if r["run_name"] == run1), None)
        run2_obj = next((r for r in runs if r["run_name"] == run2), None)
        progress.update(task, completed=True)

    if run1_obj is None or run2_obj is None:
        console.print("[red]✗ One or both runs not found[/red]")
        return

    # Compare
    console.print(
        Panel(f"[cyan]{run1}[/cyan] vs [cyan]{run2}[/cyan]", title="Run Comparison", border_style="cyan")
    )

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        task = progress.add_task("Comparing runs...", total=None)
        comparison = query.compare_runs(run1_obj["run_id"], run2_obj["run_id"], [metric])
        progress.update(task, completed=True)

    # Configuration differences
    if comparison["config_diff"]:
        console.print("\n[bold]⚙ Configuration Differences:[/bold]")

        diff_table = Table(box=box.SIMPLE)
        diff_table.add_column("Parameter", style="cyan")
        diff_table.add_column(run1, style="green")
        diff_table.add_column(run2, style="yellow")

        for key, diff in comparison["config_diff"].items():
            diff_table.add_row(key, str(diff["run1"]), str(diff["run2"]))

        console.print(diff_table)
    else:
        console.print("\n[green]✓ Identical configurations[/green]")

    # Metric comparison
    if metric in comparison["metrics_comparison"]:
        metric_data = comparison["metrics_comparison"][metric]

        console.print(f"\n[bold]📊 Metric: {metric}[/bold]")

        # Final values
        final1 = metric_data["final_value_run1"]
        final2 = metric_data["final_value_run2"]

        if final1 is not None and final2 is not None:
            console.print(f"  {run1}: {final1:.6f}")
            console.print(f"  {run2}: {final2:.6f}")

            diff = abs(final1 - final2)
            pct_diff = (diff / final1) * 100 if final1 != 0 else 0

            if final1 < final2:
                console.print(f"  [green]✓ {run1} is better by {pct_diff:.2f}%[/green]")
            else:
                console.print(f"  [green]✓ {run2} is better by {pct_diff:.2f}%[/green]")

    # Divergence analysis
    if comparison["divergence_analysis"]:
        div = comparison["divergence_analysis"]

        console.print(f"\n[bold red]⚠ Divergence Detected[/bold red]")
        console.print(f"  Step: {div['step']}")
        console.print(f"  {run1} loss: {div['loss_run1']:.6f}")
        console.print(f"  {run2} loss: {div['loss_run2']:.6f}")
        console.print(f"  Relative diff: {div['relative_diff']:.2%}")
    else:
        console.print("\n[green]✓ No significant divergence detected[/green]")


@cli.command()
@click.argument("run_name")
@click.option("--auto", is_flag=True, help="Auto-detect anomalies")
@click.option("--type", "-t", multiple=True, help="Filter by anomaly type")
@click.pass_context
def anomalies(ctx: Any, run_name: str, auto: bool, type: tuple) -> None:
    """Show or detect anomalies in a run."""
    query = ctx.obj["query"]

    # Find run
    runs = query.storage.list_runs()
    run = next((r for r in runs if r["run_name"] == run_name), None)

    if run is None:
        console.print(f"[red]✗ Run '{run_name}' not found[/red]")
        return

    # Auto-detect if requested
    if auto:
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Detecting anomalies...", total=None)
            detector = AnomalyDetector(query.storage)
            detected = detector.detect_all_anomalies(run["run_id"])
            detector.store_anomalies(detected)
            progress.update(task, completed=True)

        console.print(f"[green]✓ Detected {len(detected)} anomalies[/green]\n")

    # Get anomalies
    anomaly_types = list(type) if type else None
    found_anomalies = query.find_anomalous_checkpoints(run["run_id"], anomaly_types)

    if not found_anomalies:
        console.print("[green]✓ No anomalies detected[/green]")
        console.print("[dim]Use --auto to run anomaly detection[/dim]")
        return

    # Display anomalies
    console.print(
        Panel(
            f"[yellow]Found {len(found_anomalies)} anomalies[/yellow]",
            title=f"Anomalies: {run_name}",
            border_style="yellow",
        )
    )

    table = Table(box=box.ROUNDED)
    table.add_column("Step", justify="right", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Severity", justify="center")
    table.add_column("Details", style="dim")

    severity_colors = {"low": "green", "medium": "yellow", "high": "red", "critical": "red bold"}

    for anomaly in found_anomalies:
        details = json.loads(anomaly["details"])
        details_str = ", ".join(f"{k}={v}" for k, v in list(details.items())[:3])

        severity_color = severity_colors.get(anomaly["severity"], "white")
        severity_str = f"[{severity_color}]{anomaly['severity'].upper()}[/{severity_color}]"

        table.add_row(
            str(anomaly["step"]),
            anomaly["anomaly_type"].replace("_", " ").title(),
            severity_str,
            details_str,
        )

    console.print(table)


@cli.command()
@click.argument("run_name")
@click.pass_context
def delete(ctx: Any, run_name: str) -> None:
    """Delete a training run."""
    query = ctx.obj["query"]

    # Find run
    runs = query.storage.list_runs()
    run = next((r for r in runs if r["run_name"] == run_name), None)

    if run is None:
        console.print(f"[red]✗ Run '{run_name}' not found[/red]")
        return

    # Confirm
    if not click.confirm(f"Delete run '{run_name}'?"):
        console.print("[yellow]Cancelled[/yellow]")
        return

    # Delete from database
    query.storage.conn.execute("DELETE FROM runs WHERE run_id = ?", (run["run_id"],))
    query.storage.conn.commit()

    console.print(f"[green]✓ Deleted run '{run_name}'[/green]")


@cli.command()
@click.pass_context
def info(ctx: Any) -> None:
    """Show Tracer installation info."""
    console.print(
        Panel(
            f"[bold cyan]Tracer v{__version__}[/bold cyan]\n"
            "Deep Training Observability for PyTorch",
            title="Tracer Info",
            border_style="cyan",
        )
    )

    # Storage info
    storage_path = Path(ctx.obj["storage"])
    console.print(f"\n[bold]Storage Path:[/bold] {storage_path}")

    if storage_path.exists():
        db_path = storage_path / "tracer.db"
        if db_path.exists():
            db_size = db_path.stat().st_size / (1024**2)
            console.print(f"[bold]Database Size:[/bold] {db_size:.2f} MB")

        samples_dir = storage_path / "samples"
        if samples_dir.exists():
            sample_files = list(samples_dir.glob("*.parquet"))
            console.print(f"[bold]Sample Files:[/bold] {len(sample_files)}")
    else:
        console.print("[yellow]Storage directory does not exist[/yellow]")

    # Check dependencies
    console.print("\n[bold]Dependencies:[/bold]")

    deps = {
        "torch": "PyTorch",
        "pyarrow": "Arrow/Parquet",
        "psutil": "System Metrics",
        "GPUtil": "GPU Metrics (optional)",
        "matplotlib": "Plotting (optional)",
    }

    for module, name in deps.items():
        try:
            __import__(module)
            console.print(f"  [green]✓[/green] {name}")
        except ImportError:
            console.print(f"  [yellow]✗[/yellow] {name}")


if __name__ == "__main__":
    cli()
