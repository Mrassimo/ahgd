#!/bin/bash
set -e

echo "🚀 Setting up AHGD V3 Real Data Processing Environment"
echo "=================================================="

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y \
    build-essential \
    curl \
    git \
    htop \
    tree \
    unzip \
    wget

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install additional geospatial libraries for real boundary data
echo "🗺️  Installing geospatial libraries..."
pip install geopandas folium contextily

# Create data processing directories
echo "📁 Creating data processing directories..."
mkdir -p /tmp/ahgd_data
mkdir -p /tmp/processed_data
mkdir -p /tmp/exports

# Set up environment variables
echo "⚙️  Setting up environment..."
echo "export PYTHONPATH=/workspaces/AHGD/src" >> ~/.bashrc
echo "export AHGD_DATA_DIR=/tmp/ahgd_data" >> ~/.bashrc
echo "export POLARS_MAX_THREADS=4" >> ~/.bashrc

# Create quick-start script for real data processing
echo "📝 Creating real data processing quick-start..."
cat > /workspaces/AHGD/start_cloud_processing.sh << 'EOF'
#!/bin/bash
echo "🇦🇺 AHGD V3: Real Australian Government Data Processing"
echo "====================================================="
echo ""
echo "📊 Available Commands:"
echo "  1. Download real data:     python real_data_pipeline.py"
echo "  2. Process with Polars:    python process_real_data.py"
echo "  3. Run performance tests:  python src/performance/benchmark_suite.py"
echo "  4. Full pipeline report:   python full_pipeline_report.py"
echo ""
echo "💾 Storage locations:"
echo "  - Raw data:        /tmp/ahgd_data"
echo "  - Processed data:  /tmp/processed_data"
echo "  - Exports:         /tmp/exports"
echo ""
echo "🎯 Next step: python real_data_pipeline.py --priority=1"
echo ""
EOF

chmod +x /workspaces/AHGD/start_cloud_processing.sh

# Display environment info
echo ""
echo "✅ AHGD V3 Environment Setup Complete!"
echo "======================================"
echo "🐍 Python:     $(python --version)"
echo "📦 Pip:        $(pip --version)"
echo "🗄️  Storage:    $(df -h /tmp | tail -1 | awk '{print $4}') available in /tmp"
echo "🧠 Memory:     $(free -h | awk '/^Mem:/ {print $2}') total RAM"
echo "⚙️  CPU cores:  $(nproc) cores"
echo ""
echo "🚀 Ready to process real Australian government data!"
echo "   Run: ./start_cloud_processing.sh"
echo ""