"""
Setup script for Australian Health Geography Data (AHGD) package.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    requirements = []
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return requirements

setup(
    name="ahgd",
    version="0.1.0",
    author="AHGD Development Team",
    author_email="contact@ahgd-project.org",
    description="Australian Health Geography Data repository with robust ETL pipeline",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/ahgd",
    project_urls={
        "Bug Reports": "https://github.com/your-org/ahgd/issues",
        "Source": "https://github.com/your-org/ahgd",
        "Documentation": "https://your-org.github.io/ahgd",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.900",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autoapi>=1.8.0",
        ],
        "analysis": [
            "jupyter>=1.0.0",
            "pandas>=1.3.0",
            "geopandas>=0.10.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ahgd-extract=src.extractors.cli:main",
            "ahgd-transform=src.transformers.cli:main",
            "ahgd-validate=src.validators.cli:main",
            "ahgd-pipeline=src.pipelines.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.csv"],
    },
    zip_safe=False,
)