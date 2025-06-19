# 🏥 Australian Health Data Analytics - Data Release

## 📊 Available Datasets

### 🗄️ Complete Database
- **health_analytics.db** (5.5MB) - Complete SQLite database
- Ready to use with any SQLite client or Python sqlite3 module
- Contains 497K+ health records across 2,454 SA2 areas

### 📦 Processed Data Package  
- **processed_data.zip** (~75MB) - Clean, analysis-ready data
- Parquet format for optimal performance
- Memory optimized with 57.5% size reduction
- Includes SEIFA, health indicators, and geographic codes

### 🗺️ Geospatial Data Package
- **geospatial_data.zip** (~95MB) - SA2 boundaries and geographic data
- Multiple formats: Shapefile, GeoJSON, Parquet
- Complete Australian Statistical Areas Level 2 coverage
- GDA2020 coordinate system

### 📋 Sample Datasets
- **sample_*.csv** - Quick access sample files
- 1000 records each for testing and exploration
- Standard CSV format for easy import

## 🚀 Quick Start

### Python Usage
```python
import pandas as pd
import sqlite3

# Load SQLite database
conn = sqlite3.connect('health_analytics.db')
df = pd.read_sql_query("SELECT * FROM health_data LIMIT 1000", conn)

# Load processed data
df = pd.read_parquet('processed_data/seifa_2021_sa2.parquet')

# Load sample data  
df = pd.read_csv('sample_seifa_2021_sa2.csv')
```

### R Usage
```r
library(DBI)
library(RSQLite)

# Load SQLite database
conn <- dbConnect(RSQLite::SQLite(), "health_analytics.db")
df <- dbGetQuery(conn, "SELECT * FROM health_data LIMIT 1000")

# Load sample data
df <- read.csv("sample_seifa_2021_sa2.csv")
```

## 📚 Data Sources

- **ABS Census 2021** - Australian Bureau of Statistics
- **SEIFA 2021** - Socio-Economic Indexes for Areas  
- **SA2 Boundaries** - Australian Statistical Geography Standard
- **Health Indicators** - Derived from government health data

## 📄 License

MIT License - See LICENSE file for details.
Data usage must comply with original data source licensing.

## 🔗 Links

- **GitHub Repository**: https://github.com/Mrassimo/ahgd
- **Interactive Dashboard**: https://mrassimo.github.io/ahgd/
- **Documentation**: https://mrassimo.github.io/ahgd/docs/

---

Last Updated: 2025-06-19
