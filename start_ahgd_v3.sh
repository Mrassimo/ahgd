#!/bin/bash

# AHGD V3: Zero-Click Deployment Script
# Modern Analytics Engineering Platform - Production Ready
# Usage: ./start_ahgd_v3.sh

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose-simple.yml"
DB_VOLUME="ahgd_duckdb_volume"
HEALTH_CHECK_TIMEOUT=300  # 5 minutes

echo -e "${BLUE}🏥 AHGD V3: Modern Analytics Engineering Platform${NC}"
echo -e "${BLUE}🚀 Starting Zero-Click Deployment...${NC}"
echo ""

# Pre-flight checks
echo -e "${YELLOW}📋 Pre-flight Checks${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose not found. Please install Docker Compose first.${NC}"
    exit 1
fi

# Use docker compose (new) or docker-compose (legacy)
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

echo -e "${GREEN}✅ Docker and Docker Compose available${NC}"

# Check compose file
if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo -e "${RED}❌ Docker Compose file not found: $COMPOSE_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Compose configuration found${NC}"

# System requirements check (macOS compatible)
if command -v vm_stat &> /dev/null; then
    # macOS memory check
    AVAILABLE_MEMORY=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//' | awk '{print $1 * 4096 / 1024 / 1024}' 2>/dev/null || echo "4096")
elif command -v free &> /dev/null; then
    # Linux memory check
    AVAILABLE_MEMORY=$(free -m | awk 'NR==2{printf "%.0f", $7}' 2>/dev/null || echo "4096")
else
    # Default assumption
    AVAILABLE_MEMORY=4096
fi

if [[ ${AVAILABLE_MEMORY%.*} -lt 2048 ]]; then
    echo -e "${YELLOW}⚠️  Warning: Less than 2GB available memory. Performance may be impacted.${NC}"
fi

echo -e "${GREEN}✅ System requirements check complete${NC}"
echo ""

# Cleanup previous deployment if requested
if [[ "${1:-}" == "--clean" ]] || [[ "${1:-}" == "-c" ]]; then
    echo -e "${YELLOW}🧹 Cleaning previous deployment...${NC}"
    
    $COMPOSE_CMD -f $COMPOSE_FILE down -v --remove-orphans 2>/dev/null || true
    docker volume rm $DB_VOLUME 2>/dev/null || true
    
    echo -e "${GREEN}✅ Cleanup complete${NC}"
    echo ""
fi

# Start deployment
echo -e "${BLUE}🚀 Starting AHGD V3 Platform...${NC}"
echo ""

# Pull latest images
echo -e "${YELLOW}📥 Pulling container images...${NC}"
$COMPOSE_CMD -f $COMPOSE_FILE pull

# Build custom images
echo -e "${YELLOW}🔨 Building custom images...${NC}"
$COMPOSE_CMD -f $COMPOSE_FILE build

# Start services
echo -e "${YELLOW}🌟 Starting services...${NC}"
$COMPOSE_CMD -f $COMPOSE_FILE up -d

echo ""
echo -e "${BLUE}⏳ Waiting for services to become healthy...${NC}"

# Health check function
check_service_health() {
    local service_name=$1
    local health_url=$2
    local timeout=${3:-60}
    
    echo -n "   Checking $service_name... "
    
    for i in $(seq 1 $timeout); do
        if curl -f -s "$health_url" >/dev/null 2>&1; then
            echo -e "${GREEN}✅ Healthy${NC}"
            return 0
        fi
        sleep 1
        if [[ $((i % 10)) -eq 0 ]]; then
            echo -n "."
        fi
    done
    
    echo -e "${RED}❌ Timeout${NC}"
    return 1
}

# Wait for services to be ready
sleep 10  # Initial startup delay

# Check service health
HEALTH_CHECKS=(
    "Airflow:http://localhost:8080/health:60"
    "Streamlit:http://localhost:8501:60" 
    "FastAPI:http://localhost:8000/health:30"
    "Documentation:http://localhost:8002:30"
)

FAILED_SERVICES=()

for check in "${HEALTH_CHECKS[@]}"; do
    IFS=':' read -r service_name health_url timeout <<< "$check"
    
    if ! check_service_health "$service_name" "$health_url" "$timeout"; then
        FAILED_SERVICES+=("$service_name")
    fi
done

echo ""

# Report deployment status
if [[ ${#FAILED_SERVICES[@]} -eq 0 ]]; then
    echo -e "${GREEN}🎉 AHGD V3 Platform Successfully Deployed!${NC}"
    echo ""
    echo -e "${BLUE}📊 Access Your Analytics Platform:${NC}"
    echo -e "   🏥 ${YELLOW}Health Dashboard:${NC}     http://localhost:8501"
    echo -e "   ⚡ ${YELLOW}API Endpoint:${NC}         http://localhost:8000"
    echo -e "   🔧 ${YELLOW}Airflow (admin/admin):${NC} http://localhost:8080"
    echo -e "   📚 ${YELLOW}Documentation:${NC}        http://localhost:8002"
    echo ""
    echo -e "${GREEN}✨ Key Features Available:${NC}"
    echo -e "   • 🚀 10x faster processing with Polars + DuckDB"
    echo -e "   • 🗺️  Interactive geographic health mapping"
    echo -e "   • 📊 Real-time analytics dashboards"
    echo -e "   • 📤 Multi-format data export (CSV, Excel, Parquet, GeoJSON)"
    echo -e "   • 🔍 Drill-down from State → SA1 level"
    echo ""
    echo -e "${BLUE}💡 Quick Start:${NC}"
    echo -e "   1. Visit the Health Dashboard at http://localhost:8501"
    echo -e "   2. Select your geographic area of interest"
    echo -e "   3. Choose health indicators to explore" 
    echo -e "   4. Interactive maps and analytics await!"
    echo ""
    echo -e "${YELLOW}📖 For detailed usage, visit: http://localhost:8002${NC}"
    
else
    echo -e "${RED}⚠️  Deployment completed with issues${NC}"
    echo -e "   Failed services: ${FAILED_SERVICES[*]}"
    echo ""
    echo -e "${YELLOW}🔍 Troubleshooting:${NC}"
    echo -e "   • Check logs: $COMPOSE_CMD -f $COMPOSE_FILE logs [service-name]"
    echo -e "   • Restart failed services: $COMPOSE_CMD -f $COMPOSE_FILE restart [service-name]"
    echo -e "   • Full restart: $COMPOSE_CMD -f $COMPOSE_FILE restart"
fi

# Show running services
echo ""
echo -e "${BLUE}📋 Service Status:${NC}"
$COMPOSE_CMD -f $COMPOSE_FILE ps

echo ""
echo -e "${BLUE}🔧 Management Commands:${NC}"
echo -e "   • View logs:           $COMPOSE_CMD -f $COMPOSE_FILE logs -f"
echo -e "   • Stop platform:       $COMPOSE_CMD -f $COMPOSE_FILE down"
echo -e "   • Restart platform:    $COMPOSE_CMD -f $COMPOSE_FILE restart"
echo -e "   • Clean shutdown:      $COMPOSE_CMD -f $COMPOSE_FILE down -v"

echo ""
echo -e "${GREEN}🏥 AHGD V3: Making Australian health data as accessible as a Google search!${NC}"