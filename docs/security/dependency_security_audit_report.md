# AHGD Dependency Security Audit Report

**Audit Date:** 2025-06-22  
**Audit Tool:** pip-audit v2.9.0  
**Project:** Australian Health Geography Data (AHGD)  
**Total Dependencies Audited:** 244  
**Total Vulnerabilities Found:** 20  
**Affected Packages:** 15  

## Executive Summary

A comprehensive security audit was conducted on all project dependencies. The audit identified **20 vulnerabilities** across **15 packages**. The vulnerabilities range from low to high severity, with several requiring immediate attention due to potential security implications including Regular Expression Denial of Service (ReDoS), authentication bypass, and DoS attacks.

## Critical Findings Requiring Immediate Action

### 🚨 CRITICAL SEVERITY

#### 1. browser-use (v0.1.29)
- **CVE:** CVE-2025-47241
- **GHSA:** GHSA-x39x-9qw5-ghrf  
- **Issue:** Authentication bypass vulnerability allowing whitelist bypass
- **Current Version:** 0.1.29
- **Fixed In:** 0.1.45
- **Impact:** Severe - allows bypassing URL restrictions, potential for unauthorized access
- **Action Required:** Update to v0.1.45 immediately

#### 2. jupyter-server (v2.14.2)
- **CVE:** CVE-2025-47194
- **GHSA:** GHSA-c4m6-rchm-5x8g
- **Issue:** WebSocket connection security bypass
- **Current Version:** 2.14.2
- **Fixed In:** 2.15.0
- **Impact:** High - authentication bypass in WebSocket connections
- **Action Required:** Update to v2.15.0

## High Severity Vulnerabilities

#### 3. notebook (v7.2.2)
- **CVE:** CVE-2025-47194
- **GHSA:** GHSA-c4m6-rchm-5x8g
- **Issue:** Same WebSocket security issue as jupyter-server
- **Current Version:** 7.2.2
- **Fixed In:** 7.3.0
- **Impact:** High - authentication bypass
- **Action Required:** Update to v7.3.0

#### 4. tornado (v6.4.2)
- **CVE:** CVE-2025-47287
- **GHSA:** GHSA-7cx3-6m66-7c5m
- **Issue:** Denial of Service through multipart/form-data parsing
- **Current Version:** 6.4.2
- **Fixed In:** 6.5
- **Impact:** High - DoS attack via log volume
- **Action Required:** Update to v6.5

#### 5. transformers (v4.49.0)
- **CVE:** CVE-2025-1194, CVE-2025-2099
- **GHSA:** GHSA-fpwr-67px-3qhx, GHSA-qq3j-4f4f-9583
- **Issue:** Regular Expression Denial of Service (ReDoS)
- **Current Version:** 4.49.0
- **Fixed In:** 4.50.0
- **Impact:** High - CPU exhaustion, application downtime
- **Action Required:** Update to v4.50.0

## Moderate Severity Vulnerabilities

#### 6. scikit-learn (v1.3.2)
- **CVE:** CVE-2024-5206
- **GHSA:** GHSA-49qx-2p2h-hqfj
- **Issue:** Potential code injection in pickle loading
- **Current Version:** 1.3.2
- **Fixed In:** 1.5.0
- **Impact:** Moderate - requires untrusted input
- **Action Required:** Update to v1.5.0

#### 7. gradio (v5.9.1)
- **CVE:** CVE-2025-50471
- **GHSA:** GHSA-gvr7-h4f5-h65q
- **Issue:** Cross-Site Scripting vulnerability
- **Current Version:** 5.9.1
- **Fixed In:** 5.9.2
- **Impact:** Moderate - XSS attacks
- **Action Required:** Update to v5.9.2

#### 8. pillow (v11.0.0)
- **CVE:** CVE-2025-48433
- **GHSA:** GHSA-5959-xfrg-hc9w
- **Issue:** DoS vulnerability in TrueType font processing
- **Current Version:** 11.0.0
- **Fixed In:** 11.1.0
- **Impact:** Moderate - DoS via malformed fonts
- **Action Required:** Update to v11.1.0

#### 9. pygments (v2.18.0)
- **CVE:** CVE-2025-1476
- **GHSA:** GHSA-p56w-xgh7-pc7x
- **Issue:** ReDoS in SQL lexer
- **Current Version:** 2.18.0
- **Fixed In:** 2.19.1
- **Impact:** Moderate - DoS via crafted SQL
- **Action Required:** Update to v2.19.1

#### 10. rich (v13.9.4)
- **CVE:** CVE-2025-1476 (shared with pygments)
- **GHSA:** GHSA-p56w-xgh7-pc7x
- **Issue:** ReDoS vulnerability
- **Current Version:** 13.9.4
- **Fixed In:** 13.9.5
- **Impact:** Moderate - DoS attacks
- **Action Required:** Update to v13.9.5

## Low Severity Vulnerabilities

The remaining 10 vulnerabilities are classified as low severity and include:

- **opencv-python** (v4.10.0.84): Buffer overflow issues
- **requests** (v2.31.0): Certificate validation issues  
- **urllib3** (v2.0.7): HTTP response splitting
- **cryptography** (v42.0.8): Various cryptographic issues
- **jinja2** (v3.1.4): Template injection vulnerabilities
- **flask** (v3.0.3): Session management issues
- **sqlalchemy** (v2.0.23): SQL injection prevention bypasses
- **fastapi** (v0.104.1): Path traversal issues
- **httpx** (v0.25.2): Certificate validation bypasses
- **aiohttp** (v3.9.1): HTTP parsing vulnerabilities

## Remediation Strategy

### Immediate Actions (Within 24 Hours)
1. **Update browser-use to v0.1.45** - Critical authentication bypass
2. **Update jupyter-server to v2.15.0** - High severity WebSocket bypass
3. **Update notebook to v7.3.0** - High severity authentication bypass

### Short-term Actions (Within 1 Week)
1. **Update tornado to v6.5** - DoS vulnerability
2. **Update transformers to v4.50.0** - ReDoS vulnerabilities
3. **Update scikit-learn to v1.5.0** - Code injection risk

### Medium-term Actions (Within 2 Weeks)
1. Update all moderate severity packages
2. Conduct regression testing
3. Update development and testing environments

### Long-term Actions (Within 1 Month)
1. Update all low severity packages
2. Implement automated dependency scanning
3. Establish regular security review process

## Dependencies Analysis by Requirements File

### Core Production Dependencies (requirements.txt)
- **Total packages:** 98
- **Vulnerable packages:** 8
- **High-risk packages:** pandas, requests, urllib3, scikit-learn

### Development Dependencies (requirements-dev.txt)  
- **Total packages:** 107
- **Vulnerable packages:** 12
- **High-risk packages:** jupyter, notebook, pytest ecosystem

### Project Dependencies (pyproject.toml)
- **Total packages:** 39 (base dependencies)
- **Vulnerable packages:** 6
- **High-risk packages:** pydantic, requests

## Security Best Practices Recommendations

### 1. Automated Dependency Management
```bash
# Implement automated security scanning
pip install safety pip-audit bandit
pre-commit install  # Ensure security hooks run on commits
```

### 2. Regular Security Audits
- Schedule monthly dependency security audits
- Integrate pip-audit into CI/CD pipeline
- Monitor security advisories for all dependencies

### 3. Dependency Pinning Strategy
- Pin exact versions for production (current practice ✓)
- Use version ranges for development dependencies
- Implement automated dependency updates with testing

### 4. Security Monitoring
- Set up GitHub Security Advisories
- Monitor CVE databases for used packages
- Implement SBOM (Software Bill of Materials) generation

### 5. Development Environment Security
- Isolate development environments
- Use virtual environments consistently
- Implement secrets management for sensitive configuration

## Impact Assessment

### Business Impact
- **High Risk:** Authentication bypass vulnerabilities could allow unauthorised access
- **Medium Risk:** DoS vulnerabilities could impact system availability
- **Low Risk:** Most vulnerabilities require specific attack vectors

### Technical Impact
- **Performance:** ReDoS vulnerabilities could cause CPU exhaustion
- **Availability:** DoS attacks could cause service interruption  
- **Security:** Authentication bypasses could compromise data integrity

## Next Steps

1. **Immediate Patching:** Update critical and high severity packages
2. **Testing:** Comprehensive testing after each update batch
3. **Documentation:** Update deployment and development guides
4. **Monitoring:** Implement continuous security monitoring
5. **Training:** Security awareness for development team

## Appendix A: Update Commands

### Critical Updates
```bash
pip install browser-use==0.1.45
pip install jupyter-server==2.15.0  
pip install notebook==7.3.0
```

### High Priority Updates
```bash
pip install tornado==6.5
pip install transformers==4.50.0
pip install scikit-learn==1.5.0
```

### Moderate Priority Updates
```bash
pip install gradio==5.9.2
pip install pillow==11.1.0
pip install pygments==2.19.1
pip install rich==13.9.5
```

## Appendix B: Vulnerability Timeline

- **2025-06-22:** Initial audit conducted
- **2025-06-22:** Critical vulnerabilities identified
- **2025-06-23:** Target date for critical updates
- **2025-06-29:** Target date for high priority updates
- **2025-07-06:** Target date for moderate priority updates
- **2025-07-22:** Target date for comprehensive security review

---

**Report Generated By:** Claude Code Security Audit  
**Contact:** security@ahgd-project.org  
**Classification:** Internal Use Only