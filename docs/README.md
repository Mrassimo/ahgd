# Australian Health Analytics Dashboard Documentation

This directory contains the comprehensive documentation for the Australian Health Analytics Dashboard (AHGD) project, built with Sphinx.

## Documentation Structure

```
docs/
├── source/                 # Sphinx source files
│   ├── api/               # Auto-generated API documentation
│   ├── guides/            # User and development guides
│   ├── tutorials/         # Step-by-step tutorials
│   ├── deployment/        # Deployment documentation
│   ├── reference/         # Configuration and troubleshooting
│   ├── _static/           # Static assets (CSS, images)
│   ├── _templates/        # Custom Sphinx templates
│   └── conf.py           # Sphinx configuration
├── build/                 # Generated documentation (created during build)
├── Makefile              # Unix/Linux build commands
├── make.bat              # Windows build commands
└── README.md             # This file
```

## Quick Start

### Prerequisites

Ensure you have the documentation dependencies installed:

```bash
# Install with UV (recommended)
uv pip install -e .[dev]

# Or install specific documentation packages
uv pip install sphinx sphinx-autoapi furo myst-parser sphinx-autobuild
```

### Building Documentation

#### Using the Build Script (Recommended)

```bash
# Build HTML documentation
python scripts/build_docs.py build

# Serve documentation locally with auto-reload
python scripts/build_docs.py watch

# Run all quality checks
python scripts/build_docs.py all
```

#### Using Make Commands

**On Unix/Linux/macOS:**

```bash
cd docs

# Clean previous builds
make clean

# Build HTML documentation
make html

# Serve documentation locally
make serve

# Build with live reload
make watch

# Check for broken links
make linkcheck

# Check documentation coverage
make coverage
```

**On Windows:**

```cmd
cd docs

# Build HTML documentation
make.bat html

# Serve documentation
make.bat serve
```

### Viewing Documentation

After building, open `docs/build/html/index.html` in your browser, or use:

```bash
# Serve documentation on http://localhost:8000
python scripts/build_docs.py serve
```

## Documentation Features

### Auto-Generated API Documentation

The documentation automatically generates API reference from Python docstrings using Sphinx AutoAPI:

- **Complete API coverage** of all modules in `src/`
- **Cross-references** between modules and functions
- **Type hints** and parameter documentation
- **Usage examples** for key functions

### Modern Theme and Features

- **Furo theme** for modern, responsive design
- **Dark/light mode** toggle
- **Search functionality** with real-time results
- **Mobile-friendly** responsive layout
- **Syntax highlighting** for code blocks

### Multiple Output Formats

- **HTML** - Interactive web documentation
- **PDF** - Print-ready documentation
- **EPUB** - E-book format

### Quality Assurance

- **Link checking** to catch broken references
- **Documentation coverage** reporting
- **Spell checking** for content quality
- **Style linting** for consistent formatting

## Content Guidelines

### Writing Style

- Use **British English** spelling and conventions
- Write in **clear, concise language**
- Include **practical examples** where appropriate
- Structure content with **logical headings**

### Code Documentation

All Python code should include comprehensive docstrings:

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """
    Brief description of the function.
    
    Longer description explaining the purpose, algorithm,
    or important considerations.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2 with default value
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
        ProcessingError: When processing fails
        
    Example:
        >>> result = example_function("test", 20)
        >>> print(result)
        True
    """
```

### reStructuredText Guidelines

Use consistent reStructuredText formatting:

```rst
Page Title
==========

Section Header
--------------

Subsection Header
~~~~~~~~~~~~~~~~~

Code Blocks
^^^^^^^^^^^

.. code-block:: python

   # Python code example
   from src.config import get_config
   config = get_config()

Notes and Warnings
^^^^^^^^^^^^^^^^^^

.. note::
   This is an informational note.

.. warning::
   This is a warning about potential issues.

.. tip::
   This is a helpful tip for users.
```

## Adding New Documentation

### Creating New Pages

1. **Create the .rst file** in the appropriate directory:
   - `guides/` for user and developer guides
   - `tutorials/` for step-by-step instructions
   - `reference/` for reference material
   - `deployment/` for deployment instructions

2. **Add to table of contents** in the relevant `index.rst` file:

```rst
.. toctree::
   :maxdepth: 2

   existing_page
   your_new_page
```

3. **Follow the content guidelines** above

### Updating API Documentation

API documentation is auto-generated from docstrings. To update:

1. **Update docstrings** in the source code
2. **Rebuild documentation** to see changes
3. **Add manual examples** in `api/` files if needed

### Adding Images and Assets

1. **Add files** to `docs/source/_static/`
2. **Reference in documentation**:

```rst
.. image:: _static/your-image.png
   :alt: Description of image
   :width: 600px
```

## CI/CD Integration

### GitHub Actions

Documentation is automatically built and deployed using GitHub Actions:

- **Builds on every push** to main and develop branches
- **Deploys to GitHub Pages** on main branch
- **Runs quality checks** including link checking and spell checking
- **Tests documentation** build performance

### Quality Checks

The CI pipeline runs several quality checks:

```bash
# Documentation formatting
doc8 docs/source/

# Spell checking
codespell docs/source/

# Link checking
sphinx-build -b linkcheck docs/source/ docs/build/linkcheck/

# Coverage checking
sphinx-build -b coverage docs/source/ docs/build/coverage/
```

## Development Workflow

### Local Development

1. **Make changes** to documentation or docstrings
2. **Build and review** locally:
   ```bash
   python scripts/build_docs.py watch
   ```
3. **Run quality checks**:
   ```bash
   python scripts/build_docs.py all
   ```
4. **Commit changes** and create pull request

### Best Practices

- **Test locally** before committing
- **Check links** are not broken
- **Verify examples** work correctly
- **Update relevant sections** when changing code
- **Follow style guidelines** consistently

## Troubleshooting

### Common Issues

**Documentation Not Building:**
```bash
# Check Sphinx installation
sphinx-build --version

# Check for syntax errors
sphinx-build -W docs/source/ docs/build/html/
```

**Missing Dependencies:**
```bash
# Install all documentation dependencies
uv pip install -e .[dev]
```

**Auto-reload Not Working:**
```bash
# Try explicit autobuild command
sphinx-autobuild docs/source/ docs/build/html/
```

**Links Not Working:**
```bash
# Check for broken links
make linkcheck
```

### Performance Issues

If documentation builds slowly:

1. **Use incremental builds**: `sphinx-build -E` for environment rebuild
2. **Limit API documentation**: Adjust `autoapi_dirs` in `conf.py`
3. **Check file sizes**: Large images or assets slow builds
4. **Monitor memory usage**: Large codebases may need more memory

## Contributing

### Documentation Standards

- **Follow existing patterns** in structure and style
- **Include practical examples** in all tutorials
- **Test all code examples** to ensure they work
- **Update related sections** when making changes
- **Use cross-references** to link related concepts

### Review Process

1. **Create feature branch** for documentation changes
2. **Build and test locally** before submitting
3. **Request review** from team members
4. **Address feedback** and update as needed
5. **Merge to develop** after approval

## Support

For documentation-related questions:

- **Check existing documentation** first
- **Review troubleshooting section** above
- **Search GitHub issues** for similar problems
- **Create new issue** with specific details
- **Contact development team** for urgent issues

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Furo Theme Documentation](https://pradyunsg.me/furo/)
- [Sphinx AutoAPI](https://sphinx-autoapi.readthedocs.io/)
- [MyST Parser](https://myst-parser.readthedocs.io/)