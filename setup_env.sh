#!/bin/bash

# AHGD Environment Setup Script
# Sets up Python virtual environment and installs dependencies
# Usage: ./setup_env.sh [--dev] [--python python3.11]

set -e  # Exit on any error

# Configuration
DEFAULT_PYTHON="python3.11"
VENV_NAME="venv"
PROJECT_NAME="AHGD"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
DEV_MODE=false
PYTHON_VERSION=$DEFAULT_PYTHON

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            DEV_MODE=true
            shift
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--dev] [--python python3.11]"
            echo "  --dev      Install development dependencies"
            echo "  --python   Specify Python version (default: python3.11)"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_info "Starting $PROJECT_NAME environment setup..."
print_info "Python version: $PYTHON_VERSION"
print_info "Development mode: $DEV_MODE"

# Check if Python is available
if ! command -v $PYTHON_VERSION &> /dev/null; then
    print_error "$PYTHON_VERSION is not installed or not in PATH"
    print_info "Please install Python $PYTHON_VERSION first"
    exit 1
fi

# Check Python version
PYTHON_VERSION_OUTPUT=$($PYTHON_VERSION --version 2>&1)
print_info "Found: $PYTHON_VERSION_OUTPUT"

# Remove existing virtual environment if it exists
if [ -d "$VENV_NAME" ]; then
    print_warning "Existing virtual environment found. Removing..."
    rm -rf "$VENV_NAME"
fi

# Create virtual environment
print_info "Creating virtual environment..."
$PYTHON_VERSION -m venv "$VENV_NAME"

# Activate virtual environment
print_info "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install requirements
if [ "$DEV_MODE" = true ]; then
    print_info "Installing development dependencies..."
    pip install -r requirements-dev.txt
else
    print_info "Installing production dependencies..."
    pip install -r requirements.txt
fi

# Install the package in development mode
print_info "Installing AHGD package in development mode..."
pip install -e .

# Set up pre-commit hooks if in development mode
if [ "$DEV_MODE" = true ]; then
    print_info "Setting up pre-commit hooks..."
    pre-commit install
    pre-commit install --hook-type commit-msg
    print_success "Pre-commit hooks installed"
fi

# Create directories if they don't exist
print_info "Creating project directories..."
mkdir -p data_raw data_processed logs configs schemas

# Set up environment variables
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        print_info "Creating .env file from .env.example..."
        cp .env.example .env
        print_warning "Please edit .env file with your specific configuration"
    else
        print_warning ".env.example not found. You may need to create .env manually"
    fi
fi

# Display success message
print_success "Environment setup completed successfully!"
echo ""
print_info "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
print_info "To deactivate the environment, run:"
echo "  deactivate"
echo ""

if [ "$DEV_MODE" = true ]; then
    print_info "Development tools available:"
    echo "  - pytest: Run tests"
    echo "  - black: Format code"
    echo "  - isort: Sort imports"
    echo "  - mypy: Type checking"
    echo "  - pre-commit: Run pre-commit hooks"
    echo ""
fi

print_info "Project structure:"
echo "  - data_raw/: Raw data storage"
echo "  - data_processed/: Processed data storage"
echo "  - logs/: Application logs"
echo "  - configs/: Configuration files"
echo "  - schemas/: Data schemas"
echo ""

# Check if virtual environment is working
print_info "Verifying installation..."
python -c "import pandas, numpy, geopandas; print('Core dependencies imported successfully')"

print_success "Setup complete! Happy coding! 🚀"