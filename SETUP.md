# Setup Guide - Australian Health Data Analytics

This guide will help you set up the Australian Health Data Analytics project on your local machine.

## Prerequisites

- **Python 3.11+** - Download from [python.org](https://www.python.org/downloads/)
- **Git** - Download from [git-scm.com](https://git-scm.com/)
- **8GB+ RAM** recommended for processing large datasets
- **5GB+ free disk space** (excluding data downloads)

## Quick Setup (Recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/massimoraso/AHGD.git
cd AHGD
```

### 2. One-Command Setup
```bash
python setup_and_run.py
```

This script will:
- Install uv package manager
- Set up virtual environment  
- Install all dependencies
- Verify installation
- Launch the dashboard

### 3. Access the Dashboard
Once setup completes, the dashboard will open automatically at `http://localhost:8501`

## Manual Setup

If you prefer manual setup or the automated script encounters issues:

### 1. Install uv Package Manager
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install Dependencies
```bash
# Create and activate virtual environment with dependencies
uv sync --dev
```

### 3. Verify Installation
```bash
# Run tests to ensure everything works
python run_tests.py
```

### 4. Launch Dashboard
```bash
# Start the interactive dashboard
python run_dashboard.py
```

## Data Setup

### Sample Data (Default)
The project includes sample data for immediate use. No additional setup required.

### Full Dataset (Optional)
To download the complete 1.2GB dataset:

```bash
# Download and process all Australian government data
uv run python scripts/data_processing/download_data.py
uv run python scripts/data_processing/process_data.py
```

**Note**: Full dataset download requires ~15 minutes and 5GB temporary storage.

## Troubleshooting

### Common Issues

**Issue**: `uv: command not found`
**Solution**: 
```bash
# Add uv to PATH (restart terminal after)
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Issue**: `Permission denied` on macOS/Linux
**Solution**:
```bash
chmod +x setup_and_run.py
python setup_and_run.py
```

**Issue**: Dashboard won't start
**Solution**:
```bash
# Check if port 8501 is in use
lsof -i :8501
# Kill any process using the port, then retry
python run_dashboard.py
```

**Issue**: Tests failing
**Solution**:
```bash
# Update dependencies and retry
uv sync --dev
python run_tests.py --verbose
```

### Getting Help

1. **Check the logs**: `logs/ahgd.log` contains detailed error information
2. **Run health check**: `uv run python scripts/utils/health_check.py`
3. **View documentation**: Visit [https://massimoraso.github.io/AHGD/](https://massimoraso.github.io/AHGD/)
4. **Report issues**: Create an issue on [GitHub](https://github.com/massimoraso/AHGD/issues)

## Development Setup

For contributors and developers:

### 1. Install Development Tools
```bash
# Install with dev dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install
```

### 2. Running Tests
```bash
# Run all tests
python run_tests.py

# Run specific test categories
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v
```

### 3. Code Quality
```bash
# Run linting
uv run flake8 src/ tests/

# Run type checking  
uv run mypy src/

# Format code
uv run black src/ tests/
```

### 4. Documentation
```bash
# Build documentation locally
cd docs
uv run make html
# Open docs/build/html/index.html
```

## System Requirements

### Minimum Requirements
- **CPU**: 2+ cores
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB (5GB with full dataset)
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

### Recommended Requirements
- **CPU**: 4+ cores (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 8GB+ (16GB for large dataset processing)
- **Storage**: 10GB+ SSD space
- **Network**: Broadband connection for data downloads

## Next Steps

After successful setup:

1. **Explore the Dashboard**: Interactive health data visualisation
2. **Review Documentation**: Comprehensive guides at `/docs`
3. **Run Analysis Scripts**: Explore `/scripts` directory
4. **Check Reports**: Review analysis results in `/reports`

## Configuration

### Environment Variables
Create `.env` file for custom configuration:
```bash
cp .env.template .env
# Edit .env file with your preferences
```

### Key Settings
- `STREAMLIT_PORT`: Dashboard port (default: 8501)
- `LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `DATA_UPDATE_INTERVAL`: Auto-refresh interval for dashboards

---

**Need help?** Check our [comprehensive documentation](https://massimoraso.github.io/AHGD/) or [create an issue](https://github.com/massimoraso/AHGD/issues).