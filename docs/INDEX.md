# Documentation Index

This directory contains all documentation for the Australian Health Data Analytics project, organised into logical categories for easy navigation.

## Quick Navigation

- [**Guides**](guides/) - User and developer guides
- [**Reference**](reference/) - Technical reference materials
- [**API Documentation**](api/) - Auto-generated API documentation
- [**Assets**](assets/) - Images, HTML files, and other resources

## Documentation Structure

### Guides Directory (`guides/`)
User and developer guides for working with the system:

- `DEPLOYMENT_GUIDE.md` - Complete deployment instructions
- `OPERATIONAL_RUNBOOKS.md` - Operations and maintenance procedures
- `PRODUCTION_READINESS_CHECKLIST.md` - Pre-deployment checklist
- `CI_CD_GUIDE.md` - Continuous integration and deployment setup
- `PERFORMANCE_MONITORING_GUIDE.md` - Performance monitoring setup
- `DASHBOARD_README.md` - Dashboard-specific documentation
- `dashboard_user_guide.md` - End-user guide for the dashboard

### Reference Directory (`reference/`)
Technical reference materials and specifications:

- `REAL_DATA_SOURCES.md` - Data source documentation
- `Personal_Health_Data_Project_Plan.md` - Original project plan
- `IMMEDIATE_NEXT_STEPS.md` - Current development priorities
- `aihw_data_sources.md` - AIHW data source specifications
- `health_risk_methodology.md` - Health risk calculation methodology
- `postcode_sa2_mapping.md` - Geographic mapping reference
- `todo.md` - Development task tracking

### API Documentation (`api/`)
Auto-generated API documentation from Sphinx:

- Configuration module documentation
- Dashboard module documentation
- Performance module documentation

### Assets Directory (`assets/`)
Visual and interactive resources:

- `health_correlation_analysis.png` - Analysis visualisation
- `health_inequality_analysis.html` - Interactive analysis report
- `initial_map.html` - Geographic mapping prototype
- `interactive_health_dashboard.html` - Dashboard prototype

## Sphinx Documentation

The project uses Sphinx for comprehensive API documentation. To build the documentation:

```bash
cd docs
make html
```

The built documentation will be available in `build/html/`.

## Contributing to Documentation

When adding new documentation:

1. Place user guides in `guides/`
2. Place technical reference materials in `reference/`
3. Place images and interactive content in `assets/`
4. Update this index file to reflect new additions
5. Follow the existing naming conventions (use underscores for files, Title Case for directories)

## Documentation Standards

- Use British English spelling throughout
- Follow Markdown best practices
- Include clear headings and navigation
- Link between related documents
- Keep file names descriptive and consistent