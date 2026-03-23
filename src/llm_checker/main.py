import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich import print as rprint
from typing import Optional

app = typer.Typer(
    name="llm-checker",
    help="Find which local LLMs can run on your system.",
    add_completion=False,
)

console = Console()


def version_callback(value: bool):
    if value:
        rprint("[bold cyan]llm-checker[/bold cyan] version [green]0.1.4[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    pass


def status_style(status: str) -> tuple[str, str]:
    """Returns (icon, color) for a given status."""
    return {
        "compatible":  (">>", "green"),
        "degraded":    ("!!", "yellow"),
        "incompatible": ("xx", "red"),
    }.get(status, ("❓", "white"))


def print_results_table(results, title: str = "LLM Compatibility Results"):
    table = Table(
        title=title,
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
        show_lines=True,
    )

    table.add_column("",        width=3)
    table.add_column("Model",   style="bold white",  min_width=35)
    table.add_column("",        width=4)  # installed badge
    table.add_column("Tags",    style="dim white",   min_width=20)
    table.add_column("RAM",     justify="center",    width=8)
    table.add_column("Disk",    justify="center",    width=8)
    table.add_column("Speed",   justify="center",    width=12)
    table.add_column("Status",  min_width=30)

    for r in results:
        icon, color = status_style(r.status)
        m = r.model
        req = m.requirements

        tags_text = " ".join(f"[cyan]{t}[/cyan]" for t in m.tags)

        # Speed: prefer GPU if available, else CPU
        speed = (
            f"~{m.gpu_tokens_per_sec} tok/s"
            if m.gpu_tokens_per_sec > 0 and r.status != "incompatible"
            else f"~{m.cpu_tokens_per_sec} tok/s"
        )

        # Status column: show warnings or reasons
        if r.status == "compatible":
            status_text = Text("Ready to run", style="green")
        elif r.status == "degraded":
            status_text = Text("\n".join(f"! {w}" for w in r.warnings), style="yellow")
        else:
            status_text = Text("\n".join(f"✗ {x}" for x in r.reasons), style="red")

        installed_badge = "[green]OK[/green]" if m.installed else ""
        table.add_row(
            icon,
            m.name,
            installed_badge,
            tags_text,
            f"{req.min_ram_gb}GB",
            f"{req.min_disk_gb}GB",
            speed,
            status_text,
        )

    console.print(table)


def print_hardware_summary(hw):
    has_gpu = len(hw.gpus) > 0
    gpu_str = (
        ", ".join(f"{g.name} ({g.vram_gb}GB {g.backend.upper()})" for g in hw.gpus)
        if has_gpu else "None (CPU-only)"
    )
    avx = " ".join(filter(None, [
        "AVX"    if hw.has_avx    else "",
        "AVX2"   if hw.has_avx2   else "",
        "AVX512" if hw.has_avx512 else "",
    ])) or "None"

    summary = (
        f"[bold]OS:[/bold] {hw.os.capitalize()}  "
        f"[bold]CPU:[/bold] {hw.cpu_name}  "
        f"[bold]RAM:[/bold] {hw.available_ram_gb}/{hw.total_ram_gb}GB free  "
        f"[bold]Disk:[/bold] {hw.free_disk_gb}GB free  "
        f"[bold]GPU:[/bold] {gpu_str}  "
        f"[bold]CPU Ext:[/bold] {avx}"
    )
    console.print(Panel(summary, title="[bold cyan]Your System[/bold cyan]", border_style="cyan"))


@app.command()
def list(
    tag: Optional[str] = typer.Option(
        None, "--tag", "-t",
        help="Filter by tag: coding, general, chat, reasoning, vision, embedding",
    ),
    all: bool = typer.Option(
        False, "--all", "-a",
        help="Show incompatible models too.",
    ),
):
    """List all LLMs compatible with your system."""
    from llm_checker.hardware import get_hardware_profile
    from llm_checker.registry import load_registry, filter_by_tag, mark_installed, get_registry_source
    from llm_checker.checker import check_all
    from llm_checker.ollama import get_installed_models, ollama_is_running

    with console.status("[cyan]Scanning your system...[/cyan]"):
        hw = get_hardware_profile()
        models = load_registry()
        installed = get_installed_models()

    if installed:
        mark_installed(models, installed)
        console.print(f"[dim]ollama: {len(installed)} models installed[/dim]")

    source = get_registry_source()
    console.print(f"[dim]registry: {len(models)} models loaded from {source}[/dim]")

    if tag:
        models = filter_by_tag(models, tag)
        if not models:
            console.print(f"[red]No models found with tag:[/red] [bold]{tag}[/bold]")
            raise typer.Exit()

    print_hardware_summary(hw)

    results = check_all(models, hw)

    if not all:
        results = [r for r in results if r.status != "incompatible"]

    if not results:
        console.print("[yellow]No compatible models found for your system.[/yellow]")
        raise typer.Exit()

    title = f"LLM Compatibility — {tag.capitalize()} Models" if tag else "LLM Compatibility Results"
    print_results_table(results, title=title)

    compatible   = sum(1 for r in results if r.status == "compatible")
    degraded     = sum(1 for r in results if r.status == "degraded")
    incompatible = sum(1 for r in results if r.status == "incompatible")

    console.print(
        f"\n[green]>> {compatible} compatible[/green]  "
        f"[yellow]!! {degraded} degraded[/yellow]  "
        f"[red]xx {incompatible} incompatible[/red]\n"
    )


@app.command()
def check(
    model: str = typer.Argument(..., help="Model name or partial name to check."),
):
    """Check if a specific model is compatible with your system."""
    import questionary
    from llm_checker.hardware import get_hardware_profile
    from llm_checker.registry import load_registry, search_by_name
    from llm_checker.checker import check_compatibility

    from llm_checker.registry import get_registry_source
    with console.status("[cyan]Scanning your system...[/cyan]"):
        hw = get_hardware_profile()
        models = load_registry()
    source = get_registry_source()
    console.print(f"[dim]registry: {len(models)} models loaded from {source}[/dim]")

    matches = search_by_name(models, model)

    if not matches:
        console.print(f"[red]No models found matching:[/red] [bold]{model}[/bold]")
        raise typer.Exit()

    # If multiple matches, let user pick
    if len(matches) > 1:
        choices = [f"{m.name}  ({m.id})" for m in matches]
        selected = questionary.select(
            f"Found {len(matches)} models matching '{model}'. Pick one:",
            choices=choices,
        ).ask()

        if not selected:
            raise typer.Exit()

        idx = choices.index(selected)
        target = matches[idx]
    else:
        target = matches[0]

    print_hardware_summary(hw)

    result = check_compatibility(target, hw)
    icon, color = status_style(result.status)

    console.print(Panel.fit(
        f"{icon}  [bold]{target.name}[/bold]\n"
        f"[dim]Tags: {', '.join(target.tags)}  |  "
        f"License: {target.license}  |  "
        f"Backends: {', '.join(target.backends)}[/dim]",
        border_style=color,
        title=f"[{color}]{result.status.upper()}[/{color}]",
    ))

    if result.warnings:
        console.print("\n[yellow][bold]Warnings:[/bold][/yellow]")
        for w in result.warnings:
            console.print(f"  [yellow]! {w}[/yellow]")

    if result.reasons:
        console.print("\n[red][bold]Blockers:[/bold][/red]")
        for r in result.reasons:
            console.print(f"  [red]✗ {r}[/red]")

    if result.status != "incompatible":
        console.print(f"\n[dim]Ollama install:[/dim]  [bold cyan]ollama pull {target.ollama_name}[/bold cyan]")
        console.print(f"[dim]HuggingFace:  [/dim]  [bold cyan]https://huggingface.co/{target.huggingface_name}[/bold cyan]\n")


if __name__ == "__main__":
    app()

@app.command(name="update")
def update_registry(
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Force refresh even if cache is fresh.",
    ),
):
    """Update the model registry from remote."""
    from llm_checker.registry_updater import pull_updates, META_FILE
    import json

    with console.status("[cyan]Fetching latest model registry...[/cyan]"):
        source, count = pull_updates(force=force)

    if source == "remote_fresh":
        console.print(f"[green]Registry updated — {count} models fetched from remote.[/green]")
    elif source == "remote_cached":
        try:
            meta = json.loads(META_FILE.read_text())
            fetched_at = meta.get("fetched_at", "unknown")
            console.print(
                f"[yellow]Using cached registry ({count} models, last updated {fetched_at}).[/yellow]\n"
                f"[dim]Run with --force to refresh.[/dim]"
            )
        except Exception:
            console.print(f"[yellow]Using cached registry ({count} models).[/yellow]")
    else:
        console.print("[yellow]Could not reach remote, using bundled registry.[/yellow]")
