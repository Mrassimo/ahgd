#!/usr/bin/env python3
"""
Australian Health Analytics - Web Export Runner

Execute the comprehensive web data export process to create optimized datasets
for GitHub Pages deployment and portfolio presentation.

Usage:
    python scripts/web_export/run_web_export.py
    python scripts/web_export/run_web_export.py --output-dir custom_web_exports
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.append(str(project_root / "src"))

from web.data_export_engine import run_web_export
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export Australian Health Analytics data for web deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic export to default location
    python scripts/web_export/run_web_export.py
    
    # Export to custom directory
    python scripts/web_export/run_web_export.py --output-dir /path/to/web_assets
    
    # Specify custom data directory
    python scripts/web_export/run_web_export.py --data-dir /path/to/data --output-dir /path/to/web_assets
        """
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Source data directory (default: data)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=Path,
        default=Path("data/web_exports"),
        help="Web export output directory (default: data/web_exports)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def print_banner():
    """Print application banner."""
    banner = """
🏥 AUSTRALIAN HEALTH ANALYTICS
   Web Data Export Engine v1.0
   
   Creating optimized datasets for portfolio presentation
   GitHub Pages ready • Sub-2 second load times • 497,181+ records
    """
    
    console.print(Panel(banner, style="bold blue", border_style="blue"))


def print_export_summary(result: dict):
    """Print export summary results."""
    # Create summary table
    table = Table(title="📊 Export Summary", style="green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Export Duration", f"{result.get('export_duration_seconds', 0):.2f} seconds")
    table.add_row("Files Created", str(result.get('files_count', 0)))
    table.add_row("Success", "✅ Complete" if result.get('files_count', 0) > 0 else "❌ Failed")
    
    console.print(table)
    
    # Performance metrics
    if 'performance_metrics' in result:
        metrics_table = Table(title="⚡ Performance Metrics", style="yellow")
        metrics_table.add_column("Component", style="cyan")
        metrics_table.add_column("Achievement", style="green")
        
        for key, value in result['performance_metrics'].items():
            if isinstance(value, (int, float)):
                metrics_table.add_row(key.replace('_', ' ').title(), f"{value}")
            else:
                metrics_table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(metrics_table)


def validate_directories(data_dir: Path, output_dir: Path):
    """Validate input and output directories."""
    if not data_dir.exists():
        console.print(f"[red]❌ Data directory not found: {data_dir}[/red]")
        console.print("[yellow]💡 Run the data processing pipeline first:[/yellow]")
        console.print("   python scripts/data_pipeline/process_all_data.py")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[green]✅ Data directory: {data_dir.absolute()}[/green]")
    console.print(f"[green]✅ Output directory: {output_dir.absolute()}[/green]")


async def main():
    """Main execution function."""
    args = parse_arguments()
    
    print_banner()
    
    # Validate directories
    validate_directories(args.data_dir, args.output_dir)
    
    console.print("\n[bold]🚀 Starting web data export process...[/bold]")
    
    try:
        # Run the export
        result = await run_web_export(
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
        
        console.print("\n[bold green]🎉 Web export completed successfully![/bold green]")
        
        # Print summary
        print_export_summary(result)
        
        # Provide next steps
        console.print("\n[bold]📋 Next Steps for Portfolio Deployment:[/bold]")
        console.print(f"1. Review exported files in: [cyan]{args.output_dir}[/cyan]")
        console.print(f"2. Copy web assets to your portfolio repository")
        console.print(f"3. Deploy to GitHub Pages, Netlify, or Vercel")
        console.print(f"4. Configure web server to serve compressed .gz files")
        
        # Show key files created
        console.print(f"\n[bold]🔑 Key Files Created:[/bold]")
        key_files = [
            "geojson/sa2_boundaries/sa2_overview.geojson",
            "geojson/centroids/sa2_centroids.geojson", 
            "json/api/v1/overview.json",
            "json/dashboard/kpis.json",
            "json/statistics/health_statistics.json",
            "metadata/data_catalog.json"
        ]
        
        for file_path in key_files:
            full_path = args.output_dir / file_path
            if full_path.exists():
                size_mb = full_path.stat().st_size / (1024 * 1024)
                console.print(f"   📄 {file_path} ({size_mb:.2f} MB)")
        
        # Show manifest location
        manifest_path = args.output_dir / "export_manifest.json"
        if manifest_path.exists():
            console.print(f"\n[bold]📋 Complete file manifest: [cyan]{manifest_path}[/cyan][/bold]")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Export failed: {str(e)}[/bold red]")
        if args.verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())