"""
Command-line interface for Australian Health Analytics platform.

Modern CLI using Typer with rich formatting and async support.
"""

import asyncio
from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from data_processing.core import AustralianHealthData
from data_processing.census_processor import CensusProcessor
from data_processing.downloaders.abs_downloader import ABSDownloader

app = typer.Typer(
    name="health-analytics",
    help="Australian Health Data Analytics Platform",
    rich_markup_mode="rich"
)

console = Console()


@app.command()
def download(
    states: Optional[List[str]] = typer.Option(
        None, 
        "--states", 
        "-s",
        help="States to download data for (e.g., nsw,vic,qld)"
    ),
    data_dir: str = typer.Option(
        "data",
        "--data-dir",
        "-d", 
        help="Data directory path"
    ),
    clean: bool = typer.Option(
        False,
        "--clean",
        help="Clean old downloads before starting"
    )
):
    """
    Download essential Australian health and demographic data.
    
    Downloads ABS Census, SEIFA, and geographic boundary data
    using high-speed async downloads.
    """
    console.print("🚀 [bold blue]Australian Health Analytics Downloader[/bold blue]")
    console.print()
    
    downloader = ABSDownloader(data_dir)
    
    if clean:
        downloader.clean_old_downloads()
    
    # Convert comma-separated string to list
    if states and isinstance(states[0], str) and ',' in states[0]:
        states = states[0].split(',')
    
    async def run_download():
        results = await downloader.download_essential_data(states)
        return results
    
    results = asyncio.run(run_download())
    
    # Display results
    table = Table(title="Download Results")
    table.add_column("Dataset", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Size", style="yellow")
    
    for filename, file_path in results.items():
        if file_path and file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            table.add_row(filename, "✓ Success", f"{size_mb:.1f} MB")
        else:
            table.add_row(filename, "✗ Failed", "-")
    
    console.print(table)


@app.command()
def process(
    data_dir: str = typer.Option(
        "data",
        "--data-dir",
        "-d",
        help="Data directory path"
    ),
    skip_census: bool = typer.Option(
        False,
        "--skip-census",
        help="Skip census data processing"
    )
):
    """
    Process downloaded data using modern Polars + DuckDB pipeline.
    
    Transforms raw ABS data into analysis-ready format with
    lightning-fast processing speeds.
    """
    console.print("⚡ [bold blue]Processing Australian Health Data[/bold blue]")
    console.print()
    
    health_data = AustralianHealthData(data_dir)
    
    if not skip_census:
        # Process census data
        census_processor = CensusProcessor(data_dir)
        demographics = census_processor.process_full_pipeline()
        console.print(f"✓ Processed {len(demographics)} SA2 areas")
    
    # Run full processing pipeline
    results = health_data.lightning_fast_processing()
    
    # Display summary
    console.print()
    console.print("[bold green]Processing Complete![/bold green]")
    for key, value in results.items():
        console.print(f"  {key}: {value}")


@app.command()
def dashboard(
    data_dir: str = typer.Option(
        "data",
        "--data-dir",
        "-d",
        help="Data directory path"
    ),
    port: int = typer.Option(
        8501,
        "--port",
        "-p",
        help="Port for Streamlit dashboard"
    ),
    host: str = typer.Option(
        "localhost",
        "--host",
        help="Host for Streamlit dashboard"
    )
):
    """
    Launch interactive Streamlit dashboard.
    
    Opens web browser with health analytics dashboard
    showing population health insights and risk analysis.
    """
    import subprocess
    import sys
    
    dashboard_path = Path(__file__).parent / "web" / "streamlit" / "dashboard.py"
    
    console.print(f"🚀 [bold blue]Launching Health Analytics Dashboard[/bold blue]")
    console.print(f"   📊 Dashboard: http://{host}:{port}")
    console.print(f"   📁 Data directory: {data_dir}")
    console.print()
    
    # Set environment variable for data directory
    import os
    os.environ["HEALTH_DATA_DIR"] = str(data_dir)
    
    # Launch Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(dashboard_path),
        "--server.port", str(port),
        "--server.address", host,
        "--browser.serverAddress", host,
        "--server.headless", "false"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error launching dashboard: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status(
    data_dir: str = typer.Option(
        "data",
        "--data-dir", 
        "-d",
        help="Data directory path"
    )
):
    """
    Show status of data and processing pipeline.
    
    Displays information about downloaded datasets,
    processed files, and database contents.
    """
    data_path = Path(data_dir)
    
    console.print("📊 [bold blue]Australian Health Analytics Status[/bold blue]")
    console.print()
    
    # Check data directories
    dirs_to_check = {
        "Raw Data": data_path / "raw",
        "Processed Data": data_path / "processed", 
        "Outputs": data_path / "outputs"
    }
    
    table = Table(title="Data Directory Status")
    table.add_column("Directory", style="cyan")
    table.add_column("Exists", style="green")
    table.add_column("Files", style="yellow")
    table.add_column("Size", style="magenta")
    
    for name, dir_path in dirs_to_check.items():
        if dir_path.exists():
            files = list(dir_path.rglob("*"))
            file_count = len([f for f in files if f.is_file()])
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            size_mb = total_size / (1024 * 1024)
            
            table.add_row(
                name,
                "✓ Yes",
                str(file_count),
                f"{size_mb:.1f} MB"
            )
        else:
            table.add_row(name, "✗ No", "-", "-")
    
    console.print(table)
    
    # Check database
    db_path = data_path / "outputs" / "health_analytics.duckdb"
    if db_path.exists():
        console.print()
        console.print("🗄️  [bold green]DuckDB Database Found[/bold green]")
        
        try:
            import duckdb
            conn = duckdb.connect(str(db_path))
            
            # Check tables
            tables = conn.execute("SHOW TABLES").fetchall()
            if tables:
                console.print("   Tables:")
                for table in tables:
                    row_count = conn.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
                    console.print(f"     • {table[0]}: {row_count:,} rows")
            else:
                console.print("   No tables found")
                
            conn.close()
            
        except Exception as e:
            console.print(f"   [red]Error reading database: {e}[/red]")
    else:
        console.print()
        console.print("🗄️  [yellow]No DuckDB database found[/yellow]")
        console.print("   Run 'health-analytics process' to create database")


@app.command()
def init(
    directory: str = typer.Argument(".", help="Directory to initialize project in"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files")
):
    """
    Initialize new Australian Health Analytics project.
    
    Creates directory structure and configuration files
    for a new health analytics project.
    """
    project_path = Path(directory).resolve()
    
    console.print(f"🏗️  [bold blue]Initializing Health Analytics Project[/bold blue]")
    console.print(f"   📁 Directory: {project_path}")
    console.print()
    
    # Create directory structure
    dirs_to_create = [
        "data/raw/abs",
        "data/processed", 
        "data/outputs/json",
        "scripts",
        "docs",
        "config"
    ]
    
    for dir_path in dirs_to_create:
        full_path = project_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        console.print(f"   ✓ Created {dir_path}/")
    
    # Create .gitkeep files
    for dir_path in dirs_to_create:
        gitkeep = project_path / dir_path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    
    console.print()
    console.print("[bold green]Project initialized successfully![/bold green]")
    console.print()
    console.print("Next steps:")
    console.print("  1. health-analytics download --states nsw,vic")
    console.print("  2. health-analytics process")
    console.print("  3. health-analytics dashboard")


if __name__ == "__main__":
    app()