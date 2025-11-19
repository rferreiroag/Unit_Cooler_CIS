# HVAC Digital Twin - Maintenance Guide
## System Maintenance and Operations Manual

**Version**: 1.0
**Date**: 2025-11-18
**For**: System Administrators, DevOps Engineers, Maintenance Personnel

---

## Table of Contents

1. [Maintenance Overview](#1-maintenance-overview)
2. [Routine Maintenance](#2-routine-maintenance)
3. [Model Retraining](#3-model-retraining)
4. [System Updates](#4-system-updates)
5. [Backup & Recovery](#5-backup--recovery)
6. [Performance Monitoring](#6-performance-monitoring)
7. [Troubleshooting Procedures](#7-troubleshooting-procedures)
8. [Emergency Procedures](#8-emergency-procedures)

---

## 1. Maintenance Overview

### Maintenance Philosophy

The HVAC Digital Twin is designed for **low maintenance** and **high reliability**. This guide ensures:
- Optimal system performance
- Early problem detection
- Minimal downtime
- Long-term reliability

### Maintenance Levels

| Level | Frequency | Personnel | Duration |
|-------|-----------|-----------|----------|
| Level 1: Daily Checks | Daily | Operator | 5 min |
| Level 2: Weekly Tasks | Weekly | Technician | 30 min |
| Level 3: Monthly Review | Monthly | Engineer | 2 hours |
| Level 4: Quarterly Update | Quarterly | Data Scientist | 4 hours |
| Level 5: Annual Audit | Annually | Team | 1 day |

### Maintenance Schedule

```
Daily    ─→ Health Check, Log Review
  │
Weekly   ─→ Drift Reports, Performance Review
  │
Monthly  ─→ System Audit, Update Check
  │
Quarterly ─→ Model Retrain, Full Backup
  │
Annually ─→ Comprehensive Audit, Documentation Update
```

---

## 2. Routine Maintenance

### Daily Tasks (5 minutes)

**Performed by:** System Operator

**Task 1: System Health Check**
```bash
# Check API status
curl http://localhost:8000/health

# Expected output:
{
  "status": "healthy",
  "model_type": "lightgbm",
  "models_loaded": ["UCAOT", "UCWOT", "UCAF"],
  "version": "1.0.0"
}
```

**Task 2: Dashboard Access Verification**
- Open dashboard: `http://localhost:8501`
- Verify "✅ API Connected" status
- Make one test prediction
- Confirm results are reasonable

**Task 3: Log Review**
```bash
# Check API logs for errors
docker-compose logs --tail=100 hvac-api | grep ERROR

# Check dashboard logs
docker-compose logs --tail=100 hvac-dashboard | grep ERROR
```

**Checklist:**
- [ ] API responding (health check passes)
- [ ] Dashboard accessible
- [ ] Test prediction successful
- [ ] No ERROR logs in past 24 hours
- [ ] No alerts from monitoring system

**If any failures:** Follow [Troubleshooting Procedures](#7-troubleshooting-procedures)

### Weekly Tasks (30 minutes)

**Performed by:** System Technician

**Task 1: Drift Report Review**
```bash
# Generate drift report
python monitoring/drift_detector.py

# Review report
cat monitoring/drift_report.json

# Check for critical drift
grep -i "critical" monitoring/drift_report.json
```

**Action required if:**
- PSI > 0.2 for any feature
- Multiple features showing "WARNING"
- Any feature showing "CRITICAL"

**Task 2: Performance Metrics Review**
```bash
# Check inference times (should be <100ms)
grep "inference_time" logs/api.log | tail -100

# Check memory usage
docker stats hvac-api --no-stream

# Check disk space
df -h /var/lib/docker
```

**Task 3: Database Cleanup** (if applicable)
```bash
# Remove old logs (>30 days)
find logs/ -name "*.log" -mtime +30 -delete

# Remove old drift reports (>90 days)
find monitoring/ -name "drift_report_*.json" -mtime +90 -delete
```

**Checklist:**
- [ ] Drift report reviewed
- [ ] No critical drift detected
- [ ] Performance metrics within limits
- [ ] Disk space >20% free
- [ ] Old logs cleaned up

### Monthly Tasks (2 hours)

**Performed by:** System Engineer

**Task 1: Comprehensive System Audit**

```bash
# Run full system diagnostics
./scripts/system_audit.sh

# Check Docker container health
docker-compose ps
docker-compose logs --tail=500 > logs/monthly_audit_$(date +%Y%m%d).log
```

**Task 2: Performance Benchmarking**

```bash
# Run edge device benchmarks
python deployment/benchmarks/edge_device_benchmark.py

# Compare with baseline
# Expected: P95 latency < 0.1 ms

# Review results
cat deployment/benchmarks/benchmark_results.json
```

**Task 3: Security Updates Check**

```bash
# Check for Python package updates
pip list --outdated

# Check for Docker image updates
docker pull python:3.10-slim
docker pull eclipse-mosquitto:latest

# Review security advisories
# https://github.com/advisories
```

**Task 4: Configuration Backup**

```bash
# Backup all configuration files
tar -czf backups/config_$(date +%Y%m%d).tar.gz \
    api/ \
    dashboard/ \
    deployment/ \
    monitoring/ \
    integration/

# Verify backup
tar -tzf backups/config_$(date +%Y%m%d).tar.gz | head
```

**Checklist:**
- [ ] System audit completed
- [ ] Performance benchmarks pass
- [ ] Security updates reviewed
- [ ] Configuration backed up
- [ ] No critical issues found

### Quarterly Tasks (4 hours)

**Performed by:** Data Scientist / ML Engineer

**Task 1: Model Retraining** (see Section 3)

**Task 2: Full System Backup** (see Section 5)

**Task 3: Model Performance Validation**

```bash
# Run comprehensive evaluation
python run_sprint5_evaluation.py

# Compare with baseline performance
# Expected: R² > 0.99 for all targets

# Document any degradation
```

**Task 4: Documentation Update**
- Update technical documentation
- Review user manual for accuracy
- Update maintenance procedures if needed
- Document any configuration changes

**Checklist:**
- [ ] Model retrained (if needed)
- [ ] Full system backup completed
- [ ] Performance validated
- [ ] Documentation updated

---

## 3. Model Retraining

### When to Retrain

**Triggers for Retraining:**
1. **Drift Detection**: PSI > 0.5 for critical features
2. **Performance Degradation**: R² drops below 0.98
3. **Scheduled**: Every 3-6 months (preventive)
4. **After Equipment Changes**: New sensors, system modifications
5. **Seasonal**: Before summer/winter transition

### Retraining Procedure

**Prerequisites:**
- New training data collected (minimum 10,000 samples)
- Data quality verified
- Backup of current model completed
- Downtime window approved (1-2 hours)

**Step-by-Step Process:**

**Step 1: Data Preparation**
```bash
# Collect new data from production
# Combine with existing training data
python scripts/prepare_retraining_data.py \
    --new-data data/production/recent_data.csv \
    --output data/retraining/combined_data.csv

# Run data quality checks
python src/data/data_quality.py data/retraining/combined_data.csv
```

**Step 2: Feature Engineering**
```bash
# Apply same feature engineering as original training
python run_sprint1_pipeline.py \
    --input data/retraining/combined_data.csv \
    --output data/retraining/processed/
```

**Step 3: Model Training**
```bash
# Train new models
python scripts/retrain_models.py \
    --data data/retraining/processed/ \
    --output models/retrained/

# Expected output:
# - lightgbm_UCAOT_retrained.pkl
# - lightgbm_UCWOT_retrained.pkl
# - lightgbm_UCAF_retrained.pkl
```

**Step 4: Validation**
```bash
# Validate new models
python scripts/validate_retrained_models.py \
    --models models/retrained/ \
    --test-data data/retraining/processed/X_test.csv

# Compare with current production models
python scripts/compare_models.py \
    --old models/ \
    --new models/retrained/
```

**Step 5: Export to ONNX**
```bash
# Export new models
python deployment/onnx/export_to_onnx.py \
    --input models/retrained/ \
    --output deployment/onnx/retrained/
```

**Step 6: Deployment** (See Section 4 - System Updates)

### Retraining Validation Criteria

**Accept new model if:**
- ✅ R² ≥ current model R²
- ✅ MAPE ≤ current model MAPE
- ✅ Inference time < 100ms
- ✅ Model size < 10 MB
- ✅ Passes validation tests

**Reject and investigate if:**
- ❌ Performance degraded > 5%
- ❌ Unrealistic predictions
- ❌ Fails validation tests

### Rollback Procedure

If new model performs poorly:

```bash
# Step 1: Stop services
docker-compose down

# Step 2: Restore old models
cp models/backup/lightgbm_*.pkl models/
cp deployment/onnx/backup/*.onnx deployment/onnx/

# Step 3: Restart services
docker-compose up -d

# Step 4: Verify rollback
curl http://localhost:8000/health
```

---

## 4. System Updates

### Update Types

| Type | Frequency | Risk | Downtime |
|------|-----------|------|----------|
| Security Patch | As needed | Low | 5-15 min |
| Bug Fix | As needed | Low | 10-30 min |
| Feature Update | Monthly | Medium | 30-60 min |
| Major Version | Quarterly | High | 1-2 hours |
| Model Update | 3-6 months | Medium | 30-60 min |

### Security Update Procedure

**For Critical Security Issues:**

```bash
# Step 1: Review security advisory
# Step 2: Test patch in development environment

# Step 3: Backup current system
./scripts/backup_system.sh

# Step 4: Apply security updates
pip install --upgrade [package-name]

# Step 5: Rebuild Docker images
docker-compose build

# Step 6: Deploy with rolling update
docker-compose up -d --no-deps hvac-api

# Step 7: Verify functionality
./scripts/smoke_test.sh

# Step 8: Monitor for 24 hours
```

### Model Update Procedure

**After successful retraining:**

```bash
# Step 1: Schedule maintenance window
# Announce to users: "System maintenance 2:00-3:00 AM"

# Step 2: Backup current models
cp -r models/ models/backup_$(date +%Y%m%d)/
cp -r deployment/onnx/*.onnx deployment/onnx/backup/

# Step 3: Deploy new models
cp models/retrained/*.pkl models/
cp deployment/onnx/retrained/*.onnx deployment/onnx/

# Step 4: Restart services
docker-compose restart hvac-api

# Step 5: Smoke test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"UCWIT":7.5, "UCAIT":25.0, "UCWF":15.0, "UCAIH":50.0, "AMBT":22.0, "UCTSP":21.0}'

# Step 6: Monitor for anomalies
tail -f logs/api.log

# Step 7: Announce completion
```

### Configuration Changes

**Process:**
1. Document change in change log
2. Test in development environment
3. Backup current configuration
4. Apply change
5. Verify functionality
6. Update documentation

**Example - Changing Drift Threshold:**

```bash
# Edit configuration
nano monitoring/config.py
# Change: PSI_THRESHOLD = 0.2 → 0.25

# Restart monitoring service
docker-compose restart hvac-monitor

# Test drift detection
python monitoring/drift_detector.py --test

# Document change
echo "$(date): Changed PSI threshold to 0.25" >> CHANGELOG.md
```

---

## 5. Backup & Recovery

### Backup Strategy

**3-2-1 Rule:**
- **3** copies of data
- **2** different media types
- **1** copy offsite

### What to Backup

| Component | Frequency | Retention | Priority |
|-----------|-----------|-----------|----------|
| Models (*.pkl, *.onnx) | Daily | 90 days | CRITICAL |
| Configuration files | Daily | 30 days | HIGH |
| Application code | On change | Forever (Git) | HIGH |
| Logs | Weekly | 30 days | MEDIUM |
| Drift reports | Weekly | 90 days | MEDIUM |
| Database (if any) | Daily | 90 days | HIGH |

### Backup Procedures

**Daily Automated Backup:**

```bash
#!/bin/bash
# /scripts/daily_backup.sh

DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/daily/$DATE"

mkdir -p $BACKUP_DIR

# Backup models
cp -r models/ $BACKUP_DIR/models/
cp -r deployment/onnx/*.onnx $BACKUP_DIR/onnx/

# Backup configuration
cp -r api/ $BACKUP_DIR/api/
cp -r dashboard/ $BACKUP_DIR/dashboard/
cp -r monitoring/ $BACKUP_DIR/monitoring/

# Create archive
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR/

# Upload to remote storage (optional)
# aws s3 cp $BACKUP_DIR.tar.gz s3://hvac-backups/daily/

# Clean old backups (>30 days)
find /backups/daily/ -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_DIR.tar.gz"
```

**Weekly Full Backup:**

```bash
#!/bin/bash
# /scripts/weekly_backup.sh

DATE=$(date +%Y%m%d)
BACKUP_FILE="/backups/weekly/full_backup_$DATE.tar.gz"

# Full system backup
tar -czf $BACKUP_FILE \
    models/ \
    data/processed/ \
    deployment/ \
    api/ \
    dashboard/ \
    monitoring/ \
    integration/ \
    docs/ \
    logs/ \
    --exclude='*.pyc' \
    --exclude='__pycache__'

# Verify backup
tar -tzf $BACKUP_FILE | wc -l

# Upload to remote
# rclone copy $BACKUP_FILE remote:hvac-backups/weekly/

echo "Full backup completed: $BACKUP_FILE"
```

### Recovery Procedures

**Scenario 1: Model File Corruption**

```bash
# Identify corrupted model
python -c "import pickle; pickle.load(open('models/lightgbm_model.pkl','rb'))"
# Error: pickle.UnpicklingError

# Restore from backup
cp backups/daily/20251118/models/lightgbm_model.pkl models/

# Verify restored model
python -c "import pickle; pickle.load(open('models/lightgbm_model.pkl','rb'))"
# Success!

# Restart service
docker-compose restart hvac-api
```

**Scenario 2: Complete System Failure**

```bash
# Step 1: Identify latest valid backup
ls -lht /backups/weekly/

# Step 2: Stop services
docker-compose down

# Step 3: Extract backup
tar -xzf /backups/weekly/full_backup_20251115.tar.gz -C /tmp/restore/

# Step 4: Restore files
cp -r /tmp/restore/models/* models/
cp -r /tmp/restore/deployment/* deployment/
cp -r /tmp/restore/api/* api/
# ... (restore other components)

# Step 5: Restart services
docker-compose up -d

# Step 6: Verify all services
./scripts/smoke_test.sh

# Step 7: Check logs for errors
docker-compose logs
```

**Scenario 3: Configuration Error**

```bash
# Restore configuration from Git
git checkout HEAD -- api/config.py

# Or from backup
cp backups/daily/latest/api/config.py api/

# Restart affected services
docker-compose restart hvac-api
```

### Disaster Recovery Plan

**RTO (Recovery Time Objective)**: 2 hours
**RPO (Recovery Point Objective)**: 24 hours

**Emergency Contact List:**
1. System Administrator: [Contact]
2. Data Scientist: [Contact]
3. IT Support: [Contact]
4. Vendor Support: [Contact]

**Recovery Steps:**
1. Assess damage (30 min)
2. Locate latest backup (15 min)
3. Restore system (60 min)
4. Validate functionality (15 min)
5. Resume operations

---

## 6. Performance Monitoring

### Key Performance Indicators (KPIs)

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| API Response Time | <50ms | >100ms | >500ms |
| Model Inference (P95) | <0.1ms | >1ms | >10ms |
| Memory Usage | <500MB | >1GB | >1.5GB |
| CPU Usage | <50% | >75% | >90% |
| Disk Space | >20% free | <15% | <10% |
| Drift PSI | <0.1 | 0.1-0.2 | >0.2 |

### Monitoring Tools

**Real-time Monitoring:**
```bash
# System resources
htop

# Docker stats
docker stats

# API logs (live)
docker-compose logs -f hvac-api

# Application metrics
curl http://localhost:8000/metrics
```

**Performance Dashboards:**
- Streamlit dashboard: http://localhost:8501
- API metrics: http://localhost:8000/metrics
- Grafana (if configured): http://localhost:3000

### Setting Up Alerts

**Example - Email Alert on Drift:**

```python
# monitoring/alert_config.py

ALERT_CONFIG = {
    'drift_psi_threshold': 0.2,
    'email_enabled': True,
    'email_to': 'admin@example.com',
    'email_subject': 'HVAC Digital Twin - Drift Alert',
    'mqtt_enabled': True,
    'mqtt_topic': 'hvac/alerts/critical'
}
```

**Example - Slack Alert:**

```bash
# scripts/alert_slack.sh

WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
MESSAGE="HVAC Digital Twin Alert: Critical drift detected in UCWIT (PSI: 2.55)"

curl -X POST $WEBHOOK_URL \
  -H 'Content-Type: application/json' \
  -d "{\"text\": \"$MESSAGE\"}"
```

---

## 7. Troubleshooting Procedures

### Common Issues

**Issue 1: High Memory Usage**

**Symptoms:**
- Docker container using >1GB RAM
- System slowdown
- OOM (Out of Memory) errors

**Diagnosis:**
```bash
# Check memory usage
docker stats hvac-api --no-stream

# Check for memory leaks
docker exec hvac-api ps aux --sort=-%mem | head
```

**Solutions:**
```bash
# Restart API service
docker-compose restart hvac-api

# If persistent, increase memory limit
# Edit docker-compose.yml:
# deploy:
#   resources:
#     limits:
#       memory: 2G

docker-compose up -d
```

**Issue 2: Slow Inference (>100ms)**

**Diagnosis:**
```bash
# Check inference times
grep "inference_time" logs/api.log | tail -100

# Profile model loading
python -m cProfile -s time api/main.py
```

**Solutions:**
- Check CPU usage (may need more resources)
- Verify model files not corrupted
- Check for network latency
- Consider model optimization

**Issue 3: Drift Alerts (False Positives)**

**Symptoms:**
- Frequent drift alerts
- PSI scores fluctuating
- No actual data quality issues

**Solutions:**
```bash
# Adjust drift thresholds
nano monitoring/drift_detector.py
# Increase PSI_THRESHOLD from 0.2 to 0.3

# Re-baseline reference data
python scripts/update_reference_data.py

# Exclude seasonal features from drift detection
```

**Issue 4: API Not Responding**

**Diagnosis:**
```bash
# Check if container is running
docker-compose ps hvac-api

# Check logs
docker-compose logs --tail=50 hvac-api

# Check port binding
netstat -tlnp | grep 8000
```

**Solutions:**
```bash
# Restart API
docker-compose restart hvac-api

# If still not working, rebuild
docker-compose down
docker-compose build hvac-api
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

### Diagnostic Commands

```bash
# System health overview
./scripts/health_check.sh

# Generate diagnostic report
./scripts/generate_diagnostics.sh > diagnostics_$(date +%Y%m%d).txt

# Check all services
docker-compose ps

# View all logs
docker-compose logs --tail=1000 > all_logs.txt

# Check disk space
df -h

# Check network connectivity
ping -c 3 localhost
curl -I http://localhost:8000
```

---

## 8. Emergency Procedures

### System Down - Complete Outage

**Immediate Actions (First 5 Minutes):**

1. **Assess Situation**
   ```bash
   docker-compose ps  # Are containers running?
   docker-compose logs  # What's in the logs?
   ```

2. **Notify Stakeholders**
   - Send notification to operations team
   - Update status page (if applicable)
   - Estimate recovery time

3. **Quick Restart Attempt**
   ```bash
   docker-compose restart
   # Wait 30 seconds
   curl http://localhost:8000/health
   ```

**If Quick Restart Fails (Next 30 Minutes):**

4. **Full Restart**
   ```bash
   docker-compose down
   docker-compose up -d
   # Wait 1 minute
   ./scripts/smoke_test.sh
   ```

5. **If Still Failing - Recovery Mode**
   ```bash
   # Restore from backup
   ./scripts/emergency_restore.sh
   ```

6. **Escalate if Needed**
   - Contact senior engineer
   - Refer to disaster recovery plan

### Data Corruption Detected

**Actions:**

1. **Stop Services Immediately**
   ```bash
   docker-compose stop
   ```

2. **Isolate Corrupted Files**
   ```bash
   mv models/ models_corrupted_$(date +%Y%m%d)/
   ```

3. **Restore from Backup**
   ```bash
   cp -r backups/latest/models/ models/
   ```

4. **Verify Integrity**
   ```bash
   python scripts/verify_models.py
   ```

5. **Restart Services**
   ```bash
   docker-compose start
   ```

6. **Document Incident**
   - Record in incident log
   - Identify root cause
   - Implement preventive measures

### Security Breach Suspected

**Immediate Actions:**

1. **Isolate System**
   ```bash
   # Disable network access
   docker network disconnect bridge hvac-api
   ```

2. **Preserve Evidence**
   ```bash
   # Copy logs
   cp -r logs/ incident_logs_$(date +%Y%m%d)/
   ```

3. **Contact Security Team**

4. **Change Credentials**

5. **Audit System**

6. **Restore from Clean Backup**

---

## Maintenance Logs

### Log Templates

**Daily Checklist Log:**
```
Date: 2025-11-18
Operator: [Name]
Time: 08:00

[ ✓ ] API health check passed
[ ✓ ] Dashboard accessible
[ ✓ ] Test prediction successful
[ ✓ ] No errors in logs
[ ✓ ] No active alerts

Notes: All systems normal

Signature: __________
```

**Model Retraining Log:**
```
Date: 2025-11-18
Engineer: [Name]

Training Data:
- Samples: 50,000
- Date range: 2025-08-01 to 2025-11-01

Results:
- UCAOT R²: 0.9935 (previous: 0.9926)
- UCWOT R²: 0.9978 (previous: 0.9975)
- UCAF R²: 1.0000 (previous: 1.0000)

Decision: Deploy new models

Deployed: 2025-11-18 02:30 AM
Verified: 2025-11-18 03:00 AM

Signature: __________
```

---

## Appendices

### A. Maintenance Scripts

All maintenance scripts located in `/scripts/`:
- `health_check.sh` - Daily health verification
- `daily_backup.sh` - Automated daily backup
- `weekly_backup.sh` - Full weekly backup
- `retrain_models.py` - Model retraining
- `emergency_restore.sh` - Emergency recovery

### B. Configuration Files

Key configuration files:
- `api/config.py` - API settings
- `monitoring/drift_detector.py` - Drift thresholds
- `docker-compose.yml` - Container configuration
- `deployment/docker/Dockerfile` - Container definition

### C. Contact Information

**Internal Contacts:**
- System Administrator: [Contact]
- Data Scientist: [Contact]
- DevOps Engineer: [Contact]

**External Contacts:**
- Cloud Provider Support: [Contact]
- Hardware Vendor: [Contact]
- Security Team: [Contact]

---

**Document Revision History:**

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-11-18 | Initial release | [Name] |

**Next Review Date:** 2026-02-18 (3 months)

---

**For Emergency Support:**
📞 Call: [Emergency Number]
📧 Email: emergency@example.com
💬 Slack: #hvac-support
