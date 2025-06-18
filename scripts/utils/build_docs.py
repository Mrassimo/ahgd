#!/usr/bin/env python3
"""
Documentation build script for Australian Health Analytics Dashboard.

This script provides a convenient way to build, serve, and deploy documentation
with various options for development and production use.
"""

import argparse
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> int:
    """Run a command and return the exit code."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=False)
        return result.returncode
    except FileNotFoundError:
        print(f"Error: Command '{cmd[0]}' not found")
        return 1


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    dependencies = ['sphinx-build', 'sphinx-autobuild']
    missing = []
    
    for dep in dependencies:
        if shutil.which(dep) is None:
            missing.append(dep)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: uv pip install sphinx sphinx-autobuild")
        return False
    
    return True


def clean_build(docs_dir: Path) -> int:
    """Clean the documentation build directory."""
    build_dir = docs_dir / "build"
    if build_dir.exists():
        print(f"Cleaning {build_dir}")
        shutil.rmtree(build_dir)
    return 0


def build_html(docs_dir: Path, fast: bool = False) -> int:
    """Build HTML documentation."""
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build" / "html"
    
    cmd = ['sphinx-build', '-b', 'html']
    
    if fast:
        cmd.extend(['-E'])  # Don't use saved environment
    else:
        cmd.extend(['-a'])  # Write all files
    
    cmd.extend([str(source_dir), str(build_dir)])
    
    return run_command(cmd, cwd=docs_dir)


def build_pdf(docs_dir: Path) -> int:
    """Build PDF documentation."""
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build" / "latex"
    
    # Build LaTeX
    cmd = ['sphinx-build', '-b', 'latex', str(source_dir), str(build_dir)]
    result = run_command(cmd, cwd=docs_dir)
    
    if result != 0:
        return result
    
    # Build PDF from LaTeX
    cmd = ['make', 'all-pdf']
    return run_command(cmd, cwd=build_dir)


def serve_docs(docs_dir: Path, port: int = 8000, open_browser: bool = True) -> int:
    """Serve documentation locally."""
    html_dir = docs_dir / "build" / "html"
    
    if not html_dir.exists():
        print("HTML documentation not found. Building...")
        result = build_html(docs_dir)
        if result != 0:
            return result
    
    url = f"http://localhost:{port}"
    print(f"Serving documentation at {url}")
    
    if open_browser:
        webbrowser.open(url)
    
    cmd = ['python', '-m', 'http.server', str(port)]
    return run_command(cmd, cwd=html_dir)


def watch_docs(docs_dir: Path, port: int = 8000) -> int:
    """Watch for changes and rebuild documentation automatically."""
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build" / "html"
    
    cmd = [
        'sphinx-autobuild',
        '--host', '0.0.0.0',
        '--port', str(port),
        '--open-browser',
        str(source_dir),
        str(build_dir)
    ]
    
    return run_command(cmd, cwd=docs_dir)


def check_links(docs_dir: Path) -> int:
    """Check for broken links in documentation."""
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build" / "linkcheck"
    
    cmd = ['sphinx-build', '-b', 'linkcheck', str(source_dir), str(build_dir)]
    return run_command(cmd, cwd=docs_dir)


def check_coverage(docs_dir: Path) -> int:
    """Check documentation coverage."""
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build" / "coverage"
    
    cmd = ['sphinx-build', '-b', 'coverage', str(source_dir), str(build_dir)]
    result = run_command(cmd, cwd=docs_dir)
    
    if result == 0:
        coverage_file = build_dir / "python.txt"
        if coverage_file.exists():
            print("\nDocumentation coverage report:")
            print("=" * 50)
            with open(coverage_file) as f:
                print(f.read())
    
    return result


def spell_check(docs_dir: Path) -> int:
    """Run spell check on documentation."""
    source_dir = docs_dir / "source"
    
    # Check if codespell is available
    if shutil.which('codespell') is None:
        print("codespell not found. Install with: pip install codespell")
        return 1
    
    cmd = [
        'codespell',
        str(source_dir),
        '--skip=*.pyc,*.png,*.jpg,*.gif,*.svg',
        '--ignore-words-list=jupyter,streamlit,ahgd'
    ]
    
    return run_command(cmd)


def lint_docs(docs_dir: Path) -> int:
    """Lint documentation for style and formatting."""
    source_dir = docs_dir / "source"
    
    # Check if doc8 is available
    if shutil.which('doc8') is None:
        print("doc8 not found. Install with: pip install doc8")
        return 1
    
    cmd = ['doc8', str(source_dir)]
    return run_command(cmd)


def deploy_github_pages(docs_dir: Path) -> int:
    """Deploy documentation to GitHub Pages."""
    html_dir = docs_dir / "build" / "html"
    
    if not html_dir.exists():
        print("HTML documentation not found. Building...")
        result = build_html(docs_dir)
        if result != 0:
            return result
    
    # Check if we're in a git repository
    if not Path('.git').exists():
        print("Error: Not in a git repository")
        return 1
    
    # Create or switch to gh-pages branch
    current_branch = subprocess.check_output(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        text=True
    ).strip()
    
    commands = [
        ['git', 'checkout', 'gh-pages'],
        ['git', 'rm', '-rf', '.'],
        ['cp', '-r', f'{html_dir}/*', '.'],
        ['git', 'add', '.'],
        ['git', 'commit', '-m', 'Update documentation'],
        ['git', 'push', 'origin', 'gh-pages'],
        ['git', 'checkout', current_branch]
    ]
    
    for cmd in commands:
        result = run_command(cmd)
        if result != 0 and 'checkout gh-pages' in ' '.join(cmd):
            # Branch doesn't exist, create it
            result = run_command(['git', 'checkout', '-b', 'gh-pages'])
            if result != 0:
                return result
        elif result != 0:
            return result
    
    print("Documentation deployed to GitHub Pages")
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build and manage documentation for AHGD"
    )
    
    parser.add_argument(
        'command',
        choices=[
            'build', 'serve', 'watch', 'clean', 'pdf',
            'linkcheck', 'coverage', 'spell', 'lint',
            'deploy', 'all'
        ],
        help='Command to run'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8000,
        help='Port for serving documentation (default: 8000)'
    )
    
    parser.add_argument(
        '--fast', '-f',
        action='store_true',
        help='Fast build (don\'t rebuild everything)'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Don\'t open browser when serving'
    )
    
    args = parser.parse_args()
    
    # Find docs directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    docs_dir = project_root / "docs"
    
    if not docs_dir.exists():
        print(f"Error: Documentation directory not found: {docs_dir}")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Change to project root
    os.chdir(project_root)
    
    # Execute command
    if args.command == 'clean':
        return clean_build(docs_dir)
    
    elif args.command == 'build':
        return build_html(docs_dir, fast=args.fast)
    
    elif args.command == 'serve':
        return serve_docs(docs_dir, args.port, not args.no_browser)
    
    elif args.command == 'watch':
        return watch_docs(docs_dir, args.port)
    
    elif args.command == 'pdf':
        return build_pdf(docs_dir)
    
    elif args.command == 'linkcheck':
        return check_links(docs_dir)
    
    elif args.command == 'coverage':
        return check_coverage(docs_dir)
    
    elif args.command == 'spell':
        return spell_check(docs_dir)
    
    elif args.command == 'lint':
        return lint_docs(docs_dir)
    
    elif args.command == 'deploy':
        return deploy_github_pages(docs_dir)
    
    elif args.command == 'all':
        # Run complete documentation build and check
        commands_to_run = [
            ('clean', lambda: clean_build(docs_dir)),
            ('build', lambda: build_html(docs_dir)),
            ('linkcheck', lambda: check_links(docs_dir)),
            ('coverage', lambda: check_coverage(docs_dir)),
            ('spell', lambda: spell_check(docs_dir)),
            ('lint', lambda: lint_docs(docs_dir))
        ]
        
        for name, func in commands_to_run:
            print(f"\n{'='*50}")
            print(f"Running {name}...")
            print('='*50)
            
            result = func()
            if result != 0:
                print(f"Error in {name} step")
                return result
        
        print("\n" + "="*50)
        print("All documentation checks completed successfully!")
        print("="*50)
        return 0
    
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())