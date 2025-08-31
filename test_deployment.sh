#!/bin/bash

# AHGD V3: Test Deployment Script
# Simple test of core components without full orchestration

echo "🧪 AHGD V3 Test Deployment"
echo "=========================="

# Test 1: Core validation
echo "1. Running core validation..."
python validate_v3_implementation.py

echo ""
echo "2. Testing Streamlit app structure..."
if [ -f "streamlit_app/main.py" ]; then
    echo "✅ Streamlit app found"
    python -c "
import sys
sys.path.append('src')
sys.path.append('streamlit_app')
try:
    import streamlit_app.main as main
    print('✅ Streamlit app imports successfully')
except ImportError as e:
    print(f'⚠️  Import issue: {e}')
"
else
    echo "❌ Streamlit app not found"
fi

echo ""
echo "3. Testing basic data processing..."
python -c "
import polars as pl
import duckdb

# Test high-performance processing
data = pl.DataFrame({
    'sa1_code': [f'test_{i:06d}' for i in range(10000)],
    'health_metric': [50.0 + (i % 100) * 0.1 for i in range(10000)]
})

# Test lazy operations
result = data.lazy().with_columns([
    (pl.col('health_metric') * 1.1).alias('adjusted_metric')
]).collect()

print(f'✅ Processed {result.height} records with Polars')

# Test DuckDB
conn = duckdb.connect(':memory:')
conn.register('test_data', result.to_pandas())
query_result = conn.execute('SELECT COUNT(*) as records FROM test_data').fetchone()
print(f'✅ DuckDB query processed {query_result[0]} records')
conn.close()
"

echo ""
echo "🎉 Test deployment completed!"
echo ""
echo "Next steps:"
echo "- Install Docker Desktop for full deployment"
echo "- Or use: ./start_ahgd_v3.sh when Docker is fully ready"