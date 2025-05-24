"""Setup configuration for AHGD ETL Pipeline."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ahgd-etl",
    version="1.0.0",
    author="AHGD ETL Team",
    description="Australian Healthcare Geographic Database ETL Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/ahgd-etl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ahgd-etl=ahgd_etl.cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ahgd_etl": ["config/yaml/*.yaml"],
    },
)