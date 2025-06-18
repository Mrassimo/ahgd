# Operational Runbooks
## Australian Health Analytics Project

**Version:** 1.0  
**Date:** June 18, 2025  
**Purpose:** Production Operations Manual

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Daily Operations](#daily-operations)
3. [Incident Response](#incident-response)
4. [Maintenance Procedures](#maintenance-procedures)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Performance Monitoring](#performance-monitoring)
7. [Backup and Recovery](#backup-and-recovery)
8. [Security Operations](#security-operations)

---

## System Overview

### Architecture Components
- **Frontend:** Streamlit Dashboard (Port 8501)
- **Database:** SQLite (health_analytics.db)
- **Cache:** Redis (Port 6379)
- **Web Server:** Nginx (Ports 80/443)
- **Monitoring:** Built-in performance monitoring

### Key File Locations
```
/app/
├── health_analytics.db          # Main database
├── data/                        # Data files (1.4GB)
├── logs/                        # Application logs
├── backups/                     # Backup files
├── src/                         # Application source
└── .env                         # Configuration
```

### Service Dependencies
1. **Application** → Redis (caching)
2. **Application** → SQLite (data)
3. **Nginx** → Application (reverse proxy)
4. **Monitoring** → Application (health checks)

---

## Daily Operations

### 1. System Health Check
**Frequency:** Daily at 09:00 UTC  
**Duration:** 5 minutes  
**Responsible:** Operations Team

```bash
#!/bin/bash
# daily_health_check.sh

echo "=== Daily Health Check - $(date) ==="

# Check application status
echo "1. Checking application health..."
curl -f http://localhost:8501/_stcore/health || echo "❌ Application unhealthy"

# Check database
echo "2. Checking database..."
sqlite3 health_analytics.db "SELECT COUNT(*) FROM sqlite_master;" || echo "❌ Database error"

# Check Redis
echo "3. Checking Redis..."
redis-cli ping | grep PONG || echo "❌ Redis not responding"

# Check disk space
echo "4. Checking disk space..."
df -h | grep -E "/$|/app" | awk '{if ($5 > 90) print "❌ Disk usage high: " $5}'

# Check memory usage
echo "5. Checking memory..."
free -h | awk 'NR==2{if ($3/$2 > 0.9) print "❌ Memory usage high: " $3"/"$2}'

# Check log errors
echo "6. Checking for errors..."
tail -100 logs/ahgd.log | grep -i error | wc -l | awk '{if ($1 > 0) print "❌ " $1 " errors in last 100 log lines"}'

echo "=== Health Check Complete ==="
```

### 2. Performance Monitoring
**Frequency:** Hourly  
**Duration:** 2 minutes

```bash
#!/bin/bash
# hourly_monitoring.sh

# Check response times
echo "Response time check..."
time curl -s http://localhost:8501 > /dev/null

# Check active connections
echo "Active connections: $(netstat -an | grep :8501 | grep ESTABLISHED | wc -l)"

# Check database size
echo "Database size: $(du -h health_analytics.db | cut -f1)"

# Check cache hit rate
redis-cli info stats | grep keyspace_hits
```

### 3. Log Review
**Frequency:** Daily  
**Duration:** 10 minutes

```bash
#!/bin/bash
# log_review.sh

echo "=== Log Review - $(date) ==="

# Check for errors
echo "Errors in last 24 hours:"
grep -c "ERROR" logs/ahgd.log

# Check for warnings
echo "Warnings in last 24 hours:"
grep -c "WARNING" logs/ahgd.log

# Top error messages
echo "Top error messages:"
grep "ERROR" logs/ahgd.log | sort | uniq -c | sort -nr | head -5

# Performance alerts
echo "Performance alerts:"
grep "SLOW QUERY\|HIGH MEMORY\|TIMEOUT" logs/ahgd.log | tail -10
```

---

## Incident Response

### Severity Levels

#### Severity 1: Critical (Service Down)
- **Response Time:** 15 minutes
- **Examples:** Application completely unavailable, database corruption
- **Actions:** Immediate investigation, emergency rollback if needed

#### Severity 2: High (Degraded Performance)
- **Response Time:** 1 hour
- **Examples:** Slow response times, partial feature failure
- **Actions:** Investigation, performance optimisation

#### Severity 3: Medium (Minor Issues)
- **Response Time:** 4 hours
- **Examples:** Non-critical errors, minor UI issues
- **Actions:** Scheduled fix, monitoring

#### Severity 4: Low (Cosmetic)
- **Response Time:** Next business day
- **Examples:** Documentation updates, minor UI improvements
- **Actions:** Backlog addition

### Incident Response Procedures

#### 1. Application Down (Severity 1)
```bash
#!/bin/bash
# incident_app_down.sh

echo "=== INCIDENT: Application Down ==="

# Check container status
echo "1. Checking containers..."
docker-compose ps

# Check logs for errors
echo "2. Checking recent logs..."
docker-compose logs --tail=50 app

# Attempt restart
echo "3. Attempting restart..."
docker-compose restart app

# Wait and test
echo "4. Testing after restart..."
sleep 30
curl -f http://localhost:8501/_stcore/health

# If still down, rollback
if [ $? -ne 0 ]; then
    echo "5. Restart failed, initiating rollback..."
    # Rollback procedure here
fi
```

#### 2. Database Issues (Severity 1)
```bash
#!/bin/bash
# incident_database.sh

echo "=== INCIDENT: Database Issues ==="

# Check database integrity
echo "1. Checking database integrity..."
sqlite3 health_analytics.db "PRAGMA integrity_check;"

# Check database locks
echo "2. Checking for locks..."
lsof health_analytics.db

# Backup current database
echo "3. Creating emergency backup..."
cp health_analytics.db backups/emergency_backup_$(date +%Y%m%d_%H%M%S).db

# If corruption detected, restore from backup
if [ corruption_detected ]; then
    echo "4. Restoring from latest backup..."
    # Restore procedure here
fi
```

#### 3. Performance Issues (Severity 2)
```bash
#!/bin/bash
# incident_performance.sh

echo "=== INCIDENT: Performance Issues ==="

# Check system resources
echo "1. System resources:"
top -b -n1 | head -20

# Check memory usage
echo "2. Memory usage:"
free -h

# Check database queries
echo "3. Database performance:"
sqlite3 health_analytics.db "PRAGMA optimize;"

# Clear cache if needed
echo "4. Clearing cache..."
redis-cli FLUSHALL
```

---

## Maintenance Procedures

### 1. Scheduled Maintenance Window
**Frequency:** Monthly  
**Duration:** 2 hours  
**Window:** First Sunday 02:00-04:00 UTC

```bash
#!/bin/bash
# scheduled_maintenance.sh

echo "=== Scheduled Maintenance - $(date) ==="

# 1. Stop application
echo "1. Stopping application..."
docker-compose down

# 2. Backup database
echo "2. Creating backup..."
cp health_analytics.db backups/maintenance_backup_$(date +%Y%m%d_%H%M%S).db

# 3. Update dependencies
echo "3. Updating dependencies..."
uv pip install --upgrade -e .

# 4. Optimise database
echo "4. Optimising database..."
sqlite3 health_analytics.db "VACUUM; ANALYZE;"

# 5. Clear old logs
echo "5. Rotating logs..."
find logs/ -name "*.log" -mtime +30 -delete

# 6. Clear old backups
echo "6. Cleaning old backups..."
find backups/ -name "*.db" -mtime +90 -delete

# 7. Restart application
echo "7. Restarting application..."
docker-compose up -d

# 8. Verify restart
echo "8. Verifying restart..."
sleep 60
curl -f http://localhost:8501/_stcore/health

echo "=== Maintenance Complete ==="
```

### 2. Database Maintenance
**Frequency:** Weekly  
**Duration:** 30 minutes

```bash
#!/bin/bash
# database_maintenance.sh

echo "=== Database Maintenance - $(date) ==="

# Optimise database
echo "1. Running VACUUM..."
sqlite3 health_analytics.db "VACUUM;"

# Update statistics
echo "2. Updating statistics..."
sqlite3 health_analytics.db "ANALYZE;"

# Check integrity
echo "3. Checking integrity..."
sqlite3 health_analytics.db "PRAGMA integrity_check;"

# Compact database
echo "4. Database size before: $(du -h health_analytics.db | cut -f1)"
sqlite3 health_analytics.db "PRAGMA optimize;"
echo "5. Database size after: $(du -h health_analytics.db | cut -f1)"

echo "=== Database Maintenance Complete ==="
```

### 3. Security Updates
**Frequency:** Weekly  
**Duration:** 1 hour

```bash
#!/bin/bash
# security_updates.sh

echo "=== Security Updates - $(date) ==="

# Update system packages
echo "1. Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Update Python dependencies
echo "2. Updating Python dependencies..."
uv pip install --upgrade -e .

# Run security scan
echo "3. Running security scan..."
bandit -r src/ scripts/ > security_report_$(date +%Y%m%d).txt

# Check for vulnerabilities
echo "4. Checking for vulnerabilities..."
safety check > vulnerability_report_$(date +%Y%m%d).txt

# Update container base image
echo "5. Updating container..."
docker-compose pull
docker-compose up -d

echo "=== Security Updates Complete ==="
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Application Won't Start
**Symptoms:** Container exits immediately, no response on port 8501

**Diagnostics:**
```bash
# Check container logs
docker-compose logs app

# Check port conflicts
lsof -i :8501

# Check environment variables
docker-compose exec app env | grep -E "DATABASE|REDIS"
```

**Solutions:**
1. Check database file exists: `ls -la health_analytics.db`
2. Verify Redis connection: `redis-cli ping`
3. Check permissions: `ls -la health_analytics.db`
4. Restart with clean state: `docker-compose down && docker-compose up -d`

#### 2. Slow Performance
**Symptoms:** Long page load times, timeout errors

**Diagnostics:**
```bash
# Check system resources
top -b -n1

# Check database size
du -h health_analytics.db

# Check cache hit rate
redis-cli info stats | grep keyspace_hits

# Check slow queries
grep "SLOW" logs/ahgd.log
```

**Solutions:**
1. Clear cache: `redis-cli FLUSHALL`
2. Optimise database: `sqlite3 health_analytics.db "VACUUM; ANALYZE;"`
3. Restart application: `docker-compose restart app`
4. Check for memory leaks: Monitor memory usage over time

#### 3. Database Locked
**Symptoms:** "Database is locked" errors

**Diagnostics:**
```bash
# Check file locks
lsof health_analytics.db

# Check database integrity
sqlite3 health_analytics.db "PRAGMA integrity_check;"
```

**Solutions:**
1. Restart application: `docker-compose restart app`
2. Kill blocking processes: `fuser -k health_analytics.db`
3. Restore from backup if corrupted

#### 4. Memory Issues
**Symptoms:** Out of memory errors, application crashes

**Diagnostics:**
```bash
# Check memory usage
free -h

# Check application memory
docker stats

# Check for memory leaks
ps aux | grep streamlit
```

**Solutions:**
1. Restart application: `docker-compose restart app`
2. Increase container memory: Edit docker-compose.yml
3. Disable caching temporarily: `ENABLE_CACHING=false`
4. Optimise data processing: Review large data operations

---

## Performance Monitoring

### Key Metrics to Monitor

#### 1. Application Metrics
- Response time (target: <2 seconds)
- Error rate (target: <1%)
- Concurrent users (monitor for spikes)
- Memory usage (target: <80%)

#### 2. Database Metrics
- Query response time (target: <100ms)
- Database size growth
- Lock contention
- Cache hit ratio

#### 3. System Metrics
- CPU usage (target: <80%)
- Memory usage (target: <80%)
- Disk space (target: <90%)
- Network I/O

### Monitoring Scripts

#### 1. Real-time Monitoring
```bash
#!/bin/bash
# realtime_monitor.sh

while true; do
    clear
    echo "=== Real-time Monitoring - $(date) ==="
    
    # Application status
    echo "Application Status:"
    curl -s -w "Response time: %{time_total}s\n" http://localhost:8501/_stcore/health
    
    # System resources
    echo -e "\nSystem Resources:"
    echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
    echo "Memory: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
    echo "Disk: $(df -h / | tail -1 | awk '{print $5}')"
    
    # Database size
    echo -e "\nDatabase Size: $(du -h health_analytics.db | cut -f1)"
    
    # Active connections
    echo "Active connections: $(netstat -an | grep :8501 | grep ESTABLISHED | wc -l)"
    
    sleep 5
done
```

#### 2. Performance Report
```bash
#!/bin/bash
# performance_report.sh

echo "=== Performance Report - $(date) ==="

# Response time test
echo "1. Response Time Test:"
for i in {1..10}; do
    curl -s -w "Request $i: %{time_total}s\n" http://localhost:8501 > /dev/null
done

# Memory usage over time
echo -e "\n2. Memory Usage:"
free -h

# Database performance
echo -e "\n3. Database Performance:"
time sqlite3 health_analytics.db "SELECT COUNT(*) FROM sqlite_master;"

# Cache statistics
echo -e "\n4. Cache Statistics:"
redis-cli info stats | grep -E "keyspace_hits|keyspace_misses"

# Error analysis
echo -e "\n5. Error Analysis (last 24h):"
grep -c "ERROR" logs/ahgd.log
```

---

## Backup and Recovery

### Backup Strategy

#### 1. Automated Daily Backups
```bash
#!/bin/bash
# daily_backup.sh

BACKUP_DIR="/app/backups"
DATE=$(date +%Y%m%d_%H%M%S)

echo "=== Daily Backup - $DATE ==="

# Database backup
echo "1. Backing up database..."
sqlite3 health_analytics.db ".backup $BACKUP_DIR/health_analytics_$DATE.db"

# Configuration backup
echo "2. Backing up configuration..."
tar -czf $BACKUP_DIR/config_$DATE.tar.gz .env logging.conf

# Data backup (weekly only - large files)
if [ $(date +%u) -eq 7 ]; then
    echo "3. Weekly data backup..."
    tar -czf $BACKUP_DIR/data_$DATE.tar.gz data/processed/
fi

# Clean old backups (keep 30 days)
echo "4. Cleaning old backups..."
find $BACKUP_DIR -name "*.db" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "=== Backup Complete ==="
```

### Recovery Procedures

#### 1. Database Recovery
```bash
#!/bin/bash
# database_recovery.sh

BACKUP_DIR="/app/backups"
RECOVERY_DATE="$1"  # Format: YYYYMMDD_HHMMSS

if [ -z "$RECOVERY_DATE" ]; then
    echo "Usage: $0 YYYYMMDD_HHMMSS"
    echo "Available backups:"
    ls -la $BACKUP_DIR/health_analytics_*.db
    exit 1
fi

echo "=== Database Recovery - $RECOVERY_DATE ==="

# Stop application
echo "1. Stopping application..."
docker-compose down

# Backup current database
echo "2. Backing up current database..."
cp health_analytics.db health_analytics_pre_recovery_$(date +%Y%m%d_%H%M%S).db

# Restore from backup
echo "3. Restoring from backup..."
cp $BACKUP_DIR/health_analytics_$RECOVERY_DATE.db health_analytics.db

# Verify integrity
echo "4. Verifying integrity..."
sqlite3 health_analytics.db "PRAGMA integrity_check;"

# Restart application
echo "5. Restarting application..."
docker-compose up -d

# Test functionality
echo "6. Testing functionality..."
sleep 30
curl -f http://localhost:8501/_stcore/health

echo "=== Recovery Complete ==="
```

#### 2. Full System Recovery
```bash
#!/bin/bash
# system_recovery.sh

echo "=== Full System Recovery ==="

# Stop all services
echo "1. Stopping services..."
docker-compose down

# Restore database
echo "2. Restoring database..."
# Use latest backup
LATEST_DB=$(ls -t backups/health_analytics_*.db | head -1)
cp "$LATEST_DB" health_analytics.db

# Restore configuration
echo "3. Restoring configuration..."
LATEST_CONFIG=$(ls -t backups/config_*.tar.gz | head -1)
tar -xzf "$LATEST_CONFIG"

# Verify data integrity
echo "4. Verifying data..."
sqlite3 health_analytics.db "PRAGMA integrity_check;"

# Restart services
echo "5. Restarting services..."
docker-compose up -d

# Full system test
echo "6. Running system tests..."
sleep 60
curl -f http://localhost:8501/_stcore/health

echo "=== System Recovery Complete ==="
```

---

## Security Operations

### 1. Security Monitoring
```bash
#!/bin/bash
# security_monitoring.sh

echo "=== Security Monitoring - $(date) ==="

# Check for failed login attempts
echo "1. Failed login attempts:"
grep "Failed login" logs/ahgd.log | tail -10

# Check for suspicious activity
echo "2. Suspicious activity:"
grep -E "unusual|suspicious|attack" logs/ahgd.log | tail -10

# Check file permissions
echo "3. Critical file permissions:"
ls -la health_analytics.db .env

# Check for security updates
echo "4. Security updates available:"
apt list --upgradable 2>/dev/null | grep -i security

echo "=== Security Monitoring Complete ==="
```

### 2. Security Incident Response
```bash
#!/bin/bash
# security_incident.sh

echo "=== SECURITY INCIDENT RESPONSE ==="

# Immediate actions
echo "1. Immediate containment..."
# Block suspicious IPs (example)
# iptables -I INPUT -s suspicious_ip -j DROP

# Collect evidence
echo "2. Collecting evidence..."
cp logs/ahgd.log security_incident_$(date +%Y%m%d_%H%M%S).log

# Notify security team
echo "3. Security team notified"

# Create incident report
echo "4. Creating incident report..."
cat > security_incident_report_$(date +%Y%m%d_%H%M%S).txt << EOF
Security Incident Report
Date: $(date)
Type: [TO BE FILLED]
Description: [TO BE FILLED]
Actions Taken: [TO BE FILLED]
Status: Under Investigation
EOF

echo "=== Security Incident Response Complete ==="
```

---

## Contact Information

### Escalation Contacts

**Level 1 - Operations Team**
- Email: ops@yourcompany.com
- Phone: +1-xxx-xxx-xxxx
- Slack: #ops-team

**Level 2 - Development Team**
- Email: dev@yourcompany.com
- Phone: +1-xxx-xxx-xxxx
- Slack: #dev-team

**Level 3 - Security Team**
- Email: security@yourcompany.com
- Phone: +1-xxx-xxx-xxxx
- Slack: #security-team

### Emergency Procedures
1. **Immediate:** Contact Level 1 (Operations)
2. **15 minutes:** Escalate to Level 2 if unresolved
3. **1 hour:** Escalate to Level 3 for security issues
4. **Management:** Notify management for Severity 1 incidents

---

**Runbook Status:** ACTIVE  
**Last Updated:** June 18, 2025  
**Next Review:** July 18, 2025  
**Approved By:** Operations Team

---

*These runbooks should be tested regularly and updated based on operational experience. All procedures should be practiced during maintenance windows to ensure effectiveness.*