# Data Changelog

All notable changes to the AHGD datasets will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project's data versioning adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Pending additions to be included in next release

### Changed
- Pending changes to be included in next release

### Deprecated
- Features or data fields to be removed in future versions

### Removed
- Features or data fields removed in this release

### Fixed
- Data quality issues resolved

### Security
- Data privacy or security improvements

---

## [1.0.0] - 2025-06-20
### Added
- Initial data pipeline setup with DVC tracking
- Base directory structure:
  - `data_raw/`: Raw data storage
  - `data_processed/`: Processed data storage
- DVC configuration for data version control
- Data versioning templates:
  - `data_version.txt`: Version tracking
  - `CHANGELOG_DATA.md`: This changelog
  - `data_manifest.json`: Data inventory

### Data Sources
- [List initial data sources here]

### Schema
- [Document initial schema structure]

### Processing
- [Document initial processing steps]

---

## Template for Future Releases

## [X.Y.Z] - YYYY-MM-DD
### Added
- New data source: [source name and description]
- New fields: [field names and descriptions]
- New processing features: [feature descriptions]

### Changed
- Updated [dataset name]: [description of changes]
- Modified schema: [description of schema changes]
- Improved processing: [description of improvements]

### Deprecated
- Field [field_name] in [dataset]: Will be removed in version [X.Y.Z]
- Processing step [step_name]: Replaced by [new_method]

### Removed
- Removed [dataset/field/feature]: [reason for removal]
- Discontinued support for: [deprecated feature]

### Fixed
- Data quality issue in [dataset]: [description of fix]
- Processing bug: [description of bug and fix]

### Security
- Implemented [security measure]: [description]
- Updated privacy controls for: [dataset/field]

### Performance
- Optimized [process]: [performance improvement details]
- Reduced processing time for: [dataset] by [percentage]

### Documentation
- Added documentation for: [feature/dataset]
- Updated README for: [component]

---

## Version History Guidelines

1. **Version Numbering**:
   - MAJOR.MINOR.PATCH (e.g., 2.1.3)
   - Update MAJOR for breaking changes
   - Update MINOR for new features
   - Update PATCH for bug fixes

2. **Entry Format**:
   - Always include date in ISO format (YYYY-MM-DD)
   - Group changes by category
   - Be specific about what changed
   - Include relevant issue/ticket numbers

3. **Categories**:
   - **Added**: New features or data
   - **Changed**: Changes to existing functionality
   - **Deprecated**: Soon-to-be removed features
   - **Removed**: Removed features or data
   - **Fixed**: Bug fixes
   - **Security**: Security-related changes
   - **Performance**: Performance improvements
   - **Documentation**: Documentation updates

4. **Best Practices**:
   - Keep entries concise but informative
   - Include migration instructions for breaking changes
   - Reference related documentation
   - Tag releases in version control