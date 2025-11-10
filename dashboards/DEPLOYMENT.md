# AHGD Dashboard Deployment Guide

This guide covers deploying the AHGD Streamlit dashboard to various environments.

## Prerequisites

- Docker and Docker Compose installed
- DuckDB database (`ahgd.db`) with processed data
- Basic understanding of containerization

## Deployment Options

### 1. Local Development

**Quick Start:**
```bash
cd dashboards
pip install -r requirements.txt
streamlit run app.py
```

Access at: `http://localhost:8501`

---

### 2. Docker Deployment

#### Build and Run with Docker

```bash
# From project root
cd dashboards

# Build Docker image
docker build -t ahgd-dashboard:latest -f Dockerfile ..

# Run container (mount database)
docker run -d \
  --name ahgd-dashboard \
  -p 8501:8501 \
  -v /path/to/ahgd.db:/app/ahgd.db:ro \
  ahgd-dashboard:latest
```

Access at: `http://localhost:8501`

#### Using Docker Compose (Recommended)

```bash
# From dashboards directory
docker-compose up -d

# View logs
docker-compose logs -f dashboard

# Stop
docker-compose down
```

**Configuration:**
Edit `docker-compose.yml` to:
- Change port mappings
- Update volume paths
- Add environment variables
- Enable nginx reverse proxy

---

### 3. Streamlit Cloud Deployment

**Steps:**
1. Push code to GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select repository and branch
5. Set `dashboards/app.py` as main file
6. Add secrets if needed (see Secrets Management below)
7. Deploy!

**Limitations:**
- Free tier: Limited resources
- Database must be accessible (cloud storage or external DB)
- Public apps only (or upgrade for private)

---

### 4. AWS ECS Deployment

**Architecture:**
```
ALB → ECS Service → ECS Tasks (Dashboard) → RDS/S3 (Database)
```

**Steps:**

1. **Push image to ECR:**
```bash
# Create ECR repository
aws ecr create-repository --repository-name ahgd-dashboard

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -t ahgd-dashboard .
docker tag ahgd-dashboard:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/ahgd-dashboard:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/ahgd-dashboard:latest
```

2. **Create ECS Task Definition:**
```json
{
  "family": "ahgd-dashboard",
  "containerDefinitions": [
    {
      "name": "dashboard",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/ahgd-dashboard:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "memory": 2048,
      "cpu": 1024,
      "essential": true
    }
  ],
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "1024",
  "memory": "2048"
}
```

3. **Create ECS Service** with Application Load Balancer

4. **Configure S3 for database** (optional):
   - Store `ahgd.db` in S3
   - Mount using EFS or download on startup

---

### 5. GCP Cloud Run Deployment

**Steps:**

1. **Build and push to Google Container Registry:**
```bash
# Configure Docker for GCP
gcloud auth configure-docker

# Build image
docker build -t gcr.io/<project-id>/ahgd-dashboard:latest .

# Push to GCR
docker push gcr.io/<project-id>/ahgd-dashboard:latest
```

2. **Deploy to Cloud Run:**
```bash
gcloud run deploy ahgd-dashboard \
  --image gcr.io/<project-id>/ahgd-dashboard:latest \
  --platform managed \
  --region us-central1 \
  --port 8501 \
  --memory 2Gi \
  --cpu 2 \
  --allow-unauthenticated \
  --set-env-vars="DB_PATH=/app/ahgd.db"
```

3. **Mount database** using Cloud Storage FUSE

---

### 6. Azure Container Instances

**Steps:**

1. **Push to Azure Container Registry:**
```bash
# Create ACR
az acr create --resource-group myResourceGroup --name ahgdregistry --sku Basic

# Build and push
az acr build --registry ahgdregistry --image ahgd-dashboard:latest .
```

2. **Deploy to ACI:**
```bash
az container create \
  --resource-group myResourceGroup \
  --name ahgd-dashboard \
  --image ahgdregistry.azurecr.io/ahgd-dashboard:latest \
  --cpu 2 \
  --memory 4 \
  --registry-login-server ahgdregistry.azurecr.io \
  --registry-username <username> \
  --registry-password <password> \
  --dns-name-label ahgd-dashboard \
  --ports 8501
```

---

## Configuration

### Environment Variables

Set these in your deployment environment:

```bash
# Database
DB_PATH=/app/ahgd.db

# Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Caching
CACHE_TTL=300  # 5 minutes

# Optional: Authentication
STREAMLIT_AUTH_ENABLED=true
STREAMLIT_AUTH_USERNAME=admin
STREAMLIT_AUTH_PASSWORD=<secure-password>
```

### Secrets Management

Create `.streamlit/secrets.toml`:

```toml
[database]
path = "/secure/path/to/ahgd.db"

[authentication]
username = "admin"
password = "secure-password"

[aws]
access_key_id = "your-key"
secret_access_key = "your-secret"
region = "us-east-1"
```

**Never commit secrets to git!**

---

## SSL/HTTPS Configuration

### Using Nginx Reverse Proxy

Create `nginx.conf`:

```nginx
upstream streamlit {
    server dashboard:8501;
}

server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location / {
        proxy_pass http://streamlit;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable in `docker-compose.yml` by uncommenting nginx service.

---

## Performance Tuning

### Resource Requirements

**Minimum:**
- CPU: 1 core
- Memory: 2GB
- Storage: 5GB (+ database size)

**Recommended for Production:**
- CPU: 2-4 cores
- Memory: 4-8GB
- Storage: 20GB

### Optimization Tips

1. **Query Caching:**
   - Adjust `CACHE_TTL` in config
   - Use `@st.cache_data` decorators

2. **Database:**
   - Keep DuckDB file on fast SSD
   - Consider read replicas for high traffic

3. **Resource Limits:**
   ```yaml
   # docker-compose.yml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
       reservations:
         cpus: '1'
         memory: 2G
   ```

4. **Horizontal Scaling:**
   - Use load balancer
   - Deploy multiple containers
   - Share database via network storage

---

## Monitoring

### Health Checks

The dashboard includes built-in health checks:

```bash
# Check health
curl http://localhost:8501/_stcore/health

# Docker health check
docker inspect --format='{{.State.Health.Status}}' ahgd-dashboard
```

### Logs

```bash
# Docker logs
docker logs -f ahgd-dashboard

# Docker Compose logs
docker-compose logs -f dashboard

# Streamlit logs location (in container)
/app/.streamlit/logs/
```

### Metrics

Monitor these metrics:
- Response time
- Memory usage
- CPU utilization
- Active connections
- Error rates

Use tools like:
- Prometheus + Grafana
- AWS CloudWatch
- GCP Cloud Monitoring
- Azure Monitor

---

## Backup & Disaster Recovery

### Database Backups

```bash
# Backup DuckDB
cp ahgd.db ahgd.db.backup.$(date +%Y%m%d)

# Automated backup (cron)
0 2 * * * cp /path/to/ahgd.db /backups/ahgd.db.$(date +\%Y\%m\%d)
```

### Container Recovery

```bash
# Stop and remove
docker-compose down

# Rebuild from scratch
docker-compose up -d --build

# Restore from backup
docker cp ahgd.db.backup ahgd-dashboard:/app/ahgd.db
docker-compose restart
```

---

## Security Best Practices

1. **Authentication:**
   - Enable Streamlit authentication
   - Use OAuth for enterprise
   - Implement rate limiting

2. **Network Security:**
   - Use HTTPS only
   - Enable CORS restrictions
   - Firewall rules for port 8501

3. **Data Security:**
   - Read-only database access
   - Encrypt sensitive data
   - Regular security audits

4. **Container Security:**
   - Use official base images
   - Regular image updates
   - Scan for vulnerabilities
   - Run as non-root user

---

## Troubleshooting

### Common Issues

**Dashboard won't start:**
```bash
# Check logs
docker logs ahgd-dashboard

# Check database exists
docker exec ahgd-dashboard ls -lh /app/ahgd.db

# Test database connection
docker exec ahgd-dashboard python3 -c "import duckdb; duckdb.connect('/app/ahgd.db')"
```

**Port already in use:**
```bash
# Find process using port 8501
lsof -i :8501

# Change port in docker-compose.yml
ports:
  - "8502:8501"
```

**Out of memory:**
```bash
# Increase container memory
docker run -m 4g ahgd-dashboard

# Or in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G
```

---

## Updating the Dashboard

```bash
# Pull latest code
git pull origin main

# Rebuild Docker image
docker-compose down
docker-compose up -d --build

# Or without downtime (rolling update)
docker-compose build
docker-compose up -d --no-deps --build dashboard
```

---

## Support

For deployment issues:
1. Check logs first
2. Review this guide
3. Check Streamlit documentation
4. Contact AHGD support team

---

## Resources

- [Streamlit Deployment Docs](https://docs.streamlit.io/deploy)
- [Docker Documentation](https://docs.docker.com)
- [AWS ECS Guide](https://aws.amazon.com/ecs/)
- [GCP Cloud Run](https://cloud.google.com/run/docs)
- [Azure Container Instances](https://azure.microsoft.com/en-us/services/container-instances/)
