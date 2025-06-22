# Security Policy

## Overview

The Australian Health Geography Data (AHGD) project is committed to maintaining the highest security standards for handling Australian health and geographic data. This policy outlines our security practices, vulnerability reporting process, and response procedures.

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          | Support Status |
| ------- | ------------------ | -------------- |
| 1.0.x   | ✅ Actively Supported | Current stable release |
| 0.9.x   | ⚠️ Security fixes only | End-of-life: 2025-12-31 |
| < 0.9   | ❌ Not Supported    | Please upgrade immediately |

### Version Support Schedule

- **Current Version (1.0.x)**: Full support including security patches, bug fixes, and feature updates
- **Previous Version (0.9.x)**: Critical security fixes only until end-of-life
- **Legacy Versions**: No security support - immediate upgrade recommended

## Security Contact Information

### Primary Security Contact
- **Email**: security@ahgd-project.org
- **Response Time**: Within 24 hours for critical vulnerabilities
- **Encryption**: PGP public key available upon request

### Secondary Contacts
- **Development Team**: dev@ahgd-project.org
- **Project Maintainer**: massimo.raso@ahgd-project.org

### Emergency Contact
For critical security issues requiring immediate attention:
- **Emergency Email**: security-emergency@ahgd-project.org
- **Expected Response**: Within 2 hours during business hours (AEST)

## Vulnerability Reporting Process

### How to Report Security Vulnerabilities

**DO NOT** create public GitHub issues for security vulnerabilities. Instead:

1. **Email us directly** at security@ahgd-project.org
2. **Include the following information**:
   - Detailed description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact assessment
   - Any proof-of-concept code (if applicable)
   - Your contact information for follow-up

3. **Use encryption** if the vulnerability is sensitive:
   - Request our PGP public key
   - Encrypt sensitive details

### What to Expect After Reporting

1. **Acknowledgement**: Within 24 hours for critical issues, 48 hours for others
2. **Initial Assessment**: Within 48-72 hours
3. **Progress Updates**: Weekly updates on investigation status
4. **Resolution Timeline**: Communicated within 5 business days

## Response Time Commitments

### Critical Vulnerabilities
- **Definition**: Remote code execution, authentication bypass, data exposure
- **Acknowledgement**: Within 2 hours (business hours), 24 hours (after hours)
- **Initial Fix**: Within 24-48 hours
- **Public Disclosure**: 7-14 days after fix is available

### High Severity Vulnerabilities  
- **Definition**: Denial of service, privilege escalation, significant data integrity issues
- **Acknowledgement**: Within 24 hours
- **Initial Fix**: Within 1 week
- **Public Disclosure**: 14-30 days after fix is available

### Medium/Low Severity Vulnerabilities
- **Acknowledgement**: Within 48 hours
- **Initial Fix**: Within 2-4 weeks
- **Public Disclosure**: 30-90 days after fix is available

## Security Standards and Compliance

### Australian Government Requirements
- **Privacy Act 1988**: Full compliance with Australian privacy legislation
- **Australian Government Information Security Manual (ISM)**: Following relevant security guidelines
- **Healthcare Sector Cyber Security Framework**: Implementing recommended practices

### Data Protection Standards
- **Data at Rest**: AES-256 encryption for sensitive datasets
- **Data in Transit**: TLS 1.3 for all network communications
- **Access Control**: Role-based access with principle of least privilege
- **Audit Logging**: Comprehensive logging of all data access and modifications

### Dependency Security
- **Regular Audits**: Monthly security audits using pip-audit and GitHub Dependabot
- **Automated Scanning**: CI/CD pipeline includes security vulnerability detection
- **Update Policy**: Security patches applied within defined response timeframes

## Current Security Posture

### Recent Security Improvements (June 2025)
- **20 vulnerabilities** addressed across 15 packages
- **100% elimination** of critical and high-severity vulnerabilities
- **70% overall reduction** in security vulnerabilities
- **Comprehensive testing** of all security fixes

### Security Metrics
- ✅ **0 Critical vulnerabilities** (down from 1)
- ✅ **0 High-severity vulnerabilities** (down from 4)  
- ✅ **0 Moderate vulnerabilities** (down from 5)
- ⚠️ **6 Low-severity vulnerabilities** remaining (development tools only)

See our [Security Fix Report](docs/security/security_remediation_complete_2025-06-22.md) for complete details.

## Security Best Practices for Users

### For Developers
1. **Keep Dependencies Updated**: Regularly update to latest versions
2. **Use Virtual Environments**: Isolate project dependencies
3. **Enable Security Scanning**: Use pre-commit hooks for security checks
4. **Follow Secure Coding**: Review our security guidelines in `docs/security/`

### For Data Analysts
1. **Data Classification**: Understand sensitivity levels of different datasets
2. **Access Controls**: Use only necessary permissions for your role
3. **Secure Storage**: Store derived datasets in approved locations
4. **Incident Reporting**: Report any suspected security issues immediately

### For System Administrators
1. **Regular Updates**: Apply security patches promptly
2. **Network Security**: Implement appropriate firewall and network controls
3. **Monitoring**: Enable comprehensive security logging
4. **Backup Security**: Ensure backups are encrypted and tested

## Security Documentation

### Complete Security Documentation
- [Security Fix Report](docs/security/security_remediation_complete_2025-06-22.md)
- [Vulnerability Analysis](docs/security/detailed_vulnerability_analysis.md) 
- [Security Guidelines](docs/security/SECURITY_GUIDELINES.md)
- [Security Checklist](docs/security/SECURITY_CHECKLIST.md)

### Security Audit Reports
- [June 2025 Security Audit](docs/security/dependency_security_audit_report.md)
- [Vulnerability Summary](docs/security/vulnerabilities_summary.json)

## Incident Response Process

### Immediate Response (0-2 hours)
1. **Acknowledge** vulnerability report
2. **Assess** initial severity and impact
3. **Isolate** affected systems if necessary
4. **Assemble** response team

### Short-term Response (2-24 hours)
1. **Investigate** vulnerability thoroughly
2. **Develop** mitigation strategies
3. **Test** potential fixes
4. **Prepare** communication plan

### Medium-term Response (1-7 days)
1. **Implement** security fixes
2. **Conduct** comprehensive testing
3. **Deploy** fixes to staging environment
4. **Prepare** public disclosure

### Long-term Response (1-4 weeks)
1. **Deploy** fixes to production
2. **Monitor** for any issues
3. **Conduct** post-incident review
4. **Update** security procedures

## Security Architecture

### Defence in Depth
- **Application Layer**: Input validation, secure coding practices
- **Data Layer**: Encryption, access controls, audit trails
- **Infrastructure Layer**: Network security, system hardening
- **Operational Layer**: Security monitoring, incident response

### Key Security Components
- **Authentication & Authorisation**: Multi-factor authentication for admin access
- **Data Encryption**: End-to-end encryption for sensitive health data
- **Network Security**: VPN access for remote administration
- **Logging & Monitoring**: Comprehensive security event logging

## Responsible Disclosure

### Our Commitment
- **Acknowledge** all security reports promptly
- **Investigate** thoroughly and communicate progress
- **Fix** vulnerabilities in a reasonable timeframe  
- **Credit** security researchers appropriately (with permission)

### Security Researcher Recognition
We maintain a [Security Researchers Hall of Fame](docs/security/HALL_OF_FAME.md) to recognise contributions to project security.

### Coordinated Disclosure Timeline
- **Day 0**: Vulnerability reported
- **Day 1-2**: Initial assessment and acknowledgement
- **Day 3-7**: Investigation and fix development
- **Day 8-14**: Testing and validation
- **Day 15-30**: Deployment and public disclosure

## Security Training and Awareness

### Development Team Training
- **Secure Coding Practices**: Annual training on OWASP Top 10
- **Threat Modelling**: Regular threat assessment exercises
- **Incident Response**: Quarterly incident response drills

### User Education
- **Security Documentation**: Comprehensive security guides
- **Best Practices**: Regular communication of security best practices
- **Awareness Updates**: Monthly security awareness communications

## Compliance and Auditing

### Regular Security Audits
- **Internal Audits**: Quarterly security reviews
- **External Audits**: Annual third-party security assessments
- **Penetration Testing**: Annual penetration testing exercises

### Compliance Monitoring
- **Regulatory Compliance**: Regular compliance assessments
- **Policy Updates**: Annual review and update of security policies
- **Training Records**: Maintenance of security training records

## Contact and Escalation

### Standard Security Issues
- **Email**: security@ahgd-project.org
- **Response Time**: 24-48 hours

### Critical Security Issues
- **Email**: security-emergency@ahgd-project.org
- **Response Time**: 2-24 hours

### Non-Security Issues
- **General Support**: support@ahgd-project.org
- **Development Issues**: Create GitHub issue for non-security matters

---

**Last Updated**: 2025-06-22  
**Policy Version**: 1.0  
**Next Review Date**: 2025-12-22  

**This security policy is reviewed and updated every 6 months or after significant security incidents.**