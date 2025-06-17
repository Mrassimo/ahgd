# 🏥 Australian Health Data Analytics

**Modern data pipeline processing 497K+ Australian health records with 57.5% memory optimization**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![Records](https://img.shields.io/badge/records-497,181+-green.svg)](#)
[![Integration](https://img.shields.io/badge/integration-92.9%25-brightgreen.svg)](#)
[![Optimization](https://img.shields.io/badge/memory--optimization-57.5%25-orange.svg)](#)

## What This Does

Processes real Australian government health data (ABS Census, SEIFA, PBS) through a modern data engineering pipeline. Built as a **portfolio project** demonstrating enterprise-grade practices.

**Key Results:**
- 📊 **497,181+ records** processed from Australian government sources
- ⚡ **57.5% memory optimization** with advanced storage techniques  
- 🎯 **92.9% integration success** across SEIFA, health, and geographic datasets
- 🚀 **<2 second load times** for interactive dashboard
- 🧪 **400+ tests** across unit, integration, performance, and security domains

## Quick Start

```bash
# Clone and setup (5 minutes)
git clone https://github.com/massimoraso/australian-health-analytics.git
cd australian-health-analytics
pip install uv && uv sync

# Launch dashboard
python scripts/launch_portfolio.py
# → http://localhost:8501
```

## Tech Stack

- **Polars** - 10-30x faster than pandas
- **DuckDB** - Embedded analytics database
- **Parquet** - 60-70% compression
- **Streamlit** - Interactive dashboard
- **Bronze-Silver-Gold** - Enterprise data lake

## What's Inside

```
src/                    # Core data processing pipeline
├── data_processing/    # Real Australian data processors
├── analysis/          # Health risk & geographic analysis  
└── web/               # Interactive dashboard

tests/                 # Comprehensive testing framework
├── integration/       # End-to-end pipeline tests
├── performance/       # 1M+ record stress tests
└── security/          # Privacy compliance tests

data/                  # Data lake structure
├── bronze/           # Raw Australian government data
├── silver/           # Cleaned & validated
└── gold/             # Analytics-ready
```

## Portfolio Highlights

This project demonstrates:

**Data Engineering:**
- Modern pipeline with Polars + DuckDB
- Bronze-Silver-Gold data lake architecture
- 57.5% memory optimization techniques
- Real-time performance monitoring

**Australian Health Domain:**
- ABS Census 2021 (2,454 SA2 areas)
- SEIFA socio-economic indices  
- PBS prescription patterns
- Geographic boundary processing

**Software Engineering:**
- 400+ tests across 7 testing domains
- CI/CD pipelines and containerization
- Privacy compliance (Australian Privacy Principles)
- Production deployment readiness

## Live Demo

- **GitHub Pages**: https://mrassimo.github.io/ahgd/
- **Repository**: https://github.com/Mrassimo/ahgd

## Documentation

- [Quick Start Guide](QUICK_START.md) - 5-minute setup
- [Project Structure](PROJECT_STRUCTURE.md) - Detailed architecture
- [Phase Reports](docs/reports/) - Implementation documentation

---

**Built for the Australian health data community and career advancement in data engineering.**