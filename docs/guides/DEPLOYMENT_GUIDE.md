# Production Deployment Guide
## Australian Health Analytics Project

**Version:** 1.0  
**Date:** June 18, 2025  
**Target:** Production Environment Setup

---

## Prerequisites

### System Requirements
- **Operating System:** Linux/macOS/Windows
- **Python Version:** 3.11+
- **Memory:** Minimum 8GB RAM (16GB recommended)
- **Storage:** 10GB free space (for data and dependencies)
- **Network:** Internet access for data downloads

### Required Tools
- Docker and Docker Compose
- Git (for repository access)
- uv (Python package manager)
- Make (for automation scripts)

---

## Phase 1: Environment Setup

### 1.1 Repository Setup
```bash
# Clone repository
git clone https://github.com/your-org/australian-health-analytics.git
cd australian-health-analytics

# Verify current branch
git branch
# Should be on 'main' branch
```

### 1.2 Python Environment
```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.11

# Activate environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -e .
```

### 1.3 Dependency Verification
```bash
# Install missing critical dependencies
uv pip install folium>=0.20.0 streamlit-folium>=0.15.0

# Verify installation
python -c "import folium, streamlit_folium; print('Dependencies OK')"
```

---

## Phase 2: Data Setup

### 2.1 Data Directory Structure
```bash
# Create required directories
mkdir -p data/{raw,processed}
mkdir -p logs
mkdir -p backups

# Set permissions
chmod 755 data logs backups
```

### 2.2 Database Initialisation
```bash
# Verify database exists
ls -la health_analytics.db

# If database missing, recreate from scripts
python scripts/populate_analysis_database.py
```

### 2.3 Data Verification
```bash
# Run data verification
python verify_data.py

# Expected output: All data sources verified ✅
```

---

## Phase 3: Configuration

### 3.1 Environment Configuration
Create `.env` file in project root:
```bash
# Environment Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG_MODE=false

# Database Configuration
DATABASE_URL=sqlite:///health_analytics.db
DATABASE_POOL_SIZE=10
DATABASE_TIMEOUT=30

# Dashboard Configuration
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8501
DASHBOARD_TITLE="Australian Health Analytics"

# Performance Configuration
ENABLE_CACHING=true
CACHE_TTL=3600
ENABLE_MONITORING=true

# Security Configuration
ENABLE_AUTH=true
SECRET_KEY=your-secret-key-here
SESSION_TIMEOUT=1800

# Data Configuration
DATA_PATH=/app/data
PROCESSED_DATA_PATH=/app/data/processed
LOG_PATH=/app/logs

# External Services
REDIS_URL=redis://localhost:6379/0
MONITORING_ENDPOINT=http://localhost:9090
```

### 3.2 Logging Configuration
Create `logging.conf`:
```ini
[loggers]
keys=root,ahgd

[handlers]
keys=console,file,rotating

[formatters]
keys=detailed,simple

[logger_root]
level=INFO
handlers=console,file

[logger_ahgd]
level=INFO
handlers=rotating
qualname=ahgd
propagate=0

[handler_console]
class=StreamHandler
level=INFO
formatter=simple
args=(sys.stdout,)

[handler_file]
class=FileHandler
level=INFO
formatter=detailed
args=('logs/ahgd.log',)

[handler_rotating]
class=handlers.RotatingFileHandler
level=INFO
formatter=detailed
args=('logs/ahgd.log', 'a', 10*1024*1024, 5)

[formatter_detailed]
format=%(asctime)s %(name)s %(levelname)s %(message)s
datefmt=%Y-%m-%d %H:%M:%S

[formatter_simple]
format=%(levelname)s %(message)s
```

---

## Phase 4: Containerisation

### 4.1 Dockerfile
Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgdal-dev \
    gdal-bin \
    libspatialite7 \
    libspatialite-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv pip install --system -e .

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p data/{raw,processed} logs backups

# Set permissions
RUN chmod +x scripts/*.py

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose port
EXPOSE 8501

# Start command
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 4.2 Docker Compose
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - ENVIRONMENT=production
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./backups:/app/backups
    depends_on:
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  redis_data:
```

### 4.3 Nginx Configuration
Create `nginx.conf`:
```nginx
events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:8501;
    }

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name your-domain.com;

        ssl_certificate /etc/ssl/cert.pem;
        ssl_certificate_key /etc/ssl/key.pem;

        location / {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for Streamlit
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

---

## Phase 5: Testing and Validation

### 5.1 Pre-deployment Testing
```bash
# Run all tests
python run_tests.py --all

# Expected: All tests passing (currently failing - needs fixing)

# Run security scan
bandit -r src/ scripts/
safety check
pip-audit

# Run performance tests
python -m pytest tests/test_performance_comprehensive.py
```

### 5.2 Container Testing
```bash
# Build container
docker-compose build

# Run container tests
docker-compose up -d
docker-compose ps

# Health check
curl http://localhost:8501/_stcore/health

# Stop containers
docker-compose down
```

---

## Phase 6: Production Deployment

### 6.1 Infrastructure Setup
```bash
# Production server setup (Ubuntu/CentOS)
sudo apt-get update
sudo apt-get install docker.io docker-compose-plugin

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER
```

### 6.2 SSL Certificate Setup
```bash
# Using Let's Encrypt (recommended)
sudo apt-get install certbot
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ssl/key.pem
```

### 6.3 Production Deployment
```bash
# Deploy to production
docker-compose -f docker-compose.prod.yml up -d

# Verify deployment
docker-compose ps
docker-compose logs app

# Test production endpoint
curl https://your-domain.com
```

---

## Phase 7: Monitoring and Maintenance

### 7.1 Monitoring Setup
```bash
# Set up log rotation
sudo logrotate -d /etc/logrotate.d/ahgd

# Monitor application health
curl http://localhost:8501/_stcore/health

# Check system resources
docker stats
```

### 7.2 Backup Procedures
```bash
# Database backup
sqlite3 health_analytics.db ".backup backups/health_analytics_$(date +%Y%m%d_%H%M%S).db"

# Data backup
tar -czf backups/data_$(date +%Y%m%d_%H%M%S).tar.gz data/

# Configuration backup
tar -czf backups/config_$(date +%Y%m%d_%H%M%S).tar.gz .env logging.conf
```

### 7.3 Update Procedures
```bash
# Application updates
git pull origin main
docker-compose build
docker-compose up -d

# Dependency updates
uv pip install --upgrade -e .
```

---

## Troubleshooting

### Common Issues

#### 1. Dependency Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'folium'
# Solution:
uv pip install folium>=0.20.0 streamlit-folium>=0.15.0
```

#### 2. Database Connection Issues
```bash
# Error: Database locked or not found
# Solution:
python scripts/populate_analysis_database.py
```

#### 3. Port Already in Use
```bash
# Error: Port 8501 already in use
# Solution:
lsof -ti:8501 | xargs kill -9
# Or change port in .env file
```

#### 4. Memory Issues
```bash
# Error: Out of memory
# Solution: Increase system memory or adjust cache settings
CACHE_TTL=1800  # Reduce cache time
ENABLE_CACHING=false  # Disable caching temporarily
```

---

## Security Hardening

### 7.1 Application Security
```bash
# Enable authentication
ENABLE_AUTH=true

# Use strong secret key
SECRET_KEY=$(openssl rand -hex 32)

# Set session timeout
SESSION_TIMEOUT=1800
```

### 7.2 Container Security
```bash
# Run as non-root user (already configured in Dockerfile)
# Limit container resources
docker run --memory=2g --cpus=2 your-app

# Use security profiles
docker run --security-opt apparmor:your-profile your-app
```

### 7.3 Network Security
```bash
# Firewall configuration
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 8501/tcp  # Block direct access to app
```

---

## Performance Optimisation

### 8.1 Application Performance
```bash
# Enable caching
ENABLE_CACHING=true
CACHE_TTL=3600

# Database optimisation
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
```

### 8.2 Container Performance
```bash
# Resource limits
docker run --memory=4g --cpus=2 --memory-swap=4g your-app

# Use multi-stage builds for smaller images
# (Already configured in Dockerfile)
```

---

## Rollback Procedures

### Emergency Rollback
```bash
# Stop current deployment
docker-compose down

# Restore from backup
cp backups/health_analytics_YYYYMMDD_HHMMSS.db health_analytics.db

# Deploy previous version
git checkout previous-tag
docker-compose up -d

# Verify rollback
curl https://your-domain.com
```

---

## Maintenance Schedule

### Daily
- Monitor application logs
- Check system resources
- Verify backup completion

### Weekly
- Update dependencies (if needed)
- Run security scans
- Performance monitoring review

### Monthly
- Full system backup
- Security patch assessment
- Performance optimisation review

---

**Deployment Guide Status: DRAFT**  
**Last Updated:** June 18, 2025  
**Next Review:** July 18, 2025

---

*This guide assumes resolution of critical issues identified in the Production Readiness Checklist. Complete those items before attempting production deployment.*