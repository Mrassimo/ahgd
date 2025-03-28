"""
Convert Python scripts to Jupyter notebooks using jupytext.
"""

import subprocess
import sys
from pathlib import Path

def install_jupytext():
    """Install jupytext package if not already installed."""
    try:
        import jupytext
        print("jupytext already installed.")
    except ImportError:
        print("Installing jupytext...")
        subprocess.run([sys.executable, "-m", "pip", "install", "jupytext"])

def convert_to_notebook(py_file):
    """Convert a Python file to Jupyter notebook."""
    import jupytext
    
    py_path = Path(py_file)
    if not py_path.exists():
        print(f"Error: {py_file} not found!")
        return False
    
    notebook = jupytext.read(py_file)
    ipynb_file = py_path.with_suffix('.ipynb')
    jupytext.write(notebook, ipynb_file)
    print(f"Created notebook: {ipynb_file}")
    return True

def main():
    """Convert all Python scripts to notebooks."""
    print("Starting conversion process...")
    
    # Install jupytext if needed
    install_jupytext()
    
    # Files to convert
    files_to_convert = [
        'colab_connect.py',
        'ahgd_etl_notebook.py'
    ]
    
    # Convert each file
    for file in files_to_convert:
        print(f"\nConverting {file}...")
        if convert_to_notebook(file):
            print(f"Successfully converted {file} to notebook format.")
        else:
            print(f"Failed to convert {file}.")
    
    print("\nConversion process complete!")

if __name__ == "__main__":
    main() 