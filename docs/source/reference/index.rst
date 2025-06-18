Reference Documentation
=======================

This section provides reference information for configuration, troubleshooting,
and advanced topics.

.. toctree::
   :maxdepth: 2

   configuration
   data_sources
   troubleshooting
   faq
   glossary

Reference Overview
------------------

The reference section contains:

**Configuration Reference**
  Complete configuration options and environment variables

**Data Sources**
  Detailed information about Australian health data sources

**Troubleshooting**
  Solutions to common problems and error messages

**FAQ**
  Frequently asked questions and answers

**Glossary**
  Definitions of terms and concepts used in the platform

Quick Reference
---------------

**Configuration Files**
  * ``pyproject.toml`` - Project configuration
  * ``.env`` - Environment variables
  * ``src/config.py`` - Application configuration

**Data Directories**
  * ``data/raw/`` - Original downloaded data
  * ``data/processed/`` - Cleaned and processed data
  * ``logs/`` - Application logs

**Key Commands**
  * ``python run_dashboard.py`` - Start dashboard
  * ``python scripts/download_data.py`` - Download data
  * ``python -m pytest`` - Run tests
  * ``ruff check src/`` - Code quality checks

Support Resources
-----------------

* :doc:`troubleshooting` - Common issues and solutions
* :doc:`faq` - Frequently asked questions
* :doc:`../guides/index` - User and developer guides
* Project repository - Source code and issue tracking