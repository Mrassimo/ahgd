"""
Configuration settings for AHGD Dashboard.
"""

from pathlib import Path

# Database configuration
DB_PATH = Path(__file__).parent.parent / "ahgd.db"

# Dashboard configuration
DASHBOARD_TITLE = "Australian Health Geography Data"
DASHBOARD_ICON = "🏥"
PAGE_LAYOUT = "wide"

# Color schemes
HEALTH_COLOR_SCALE = [
    "#d73027",  # Red (poor health)
    "#fc8d59",
    "#fee08b",
    "#d9ef8b",
    "#91cf60",
    "#1a9850",  # Green (good health)
]

SEIFA_COLOR_SCALE = [
    "#8c510a",  # Brown (disadvantaged)
    "#d8b365",
    "#f6e8c3",
    "#c7eae5",
    "#5ab4ac",
    "#01665e",  # Teal (advantaged)
]

# Map configuration
MAP_CENTER = [-25.2744, 133.7751]  # Australia center
MAP_ZOOM = 4

# Cache configuration (in seconds)
CACHE_TTL = 300  # 5 minutes

# Data freshness thresholds
DATA_FRESH_HOURS = 24
DATA_STALE_HOURS = 72

# Metrics configuration
METRICS_CONFIG = {
    "mortality_rate": {
        "label": "Mortality Rate",
        "format": "{:.2f}",
        "description": "Deaths per 1,000 population",
        "good_threshold": 5.0,
        "bad_threshold": 10.0,
    },
    "utilisation_rate": {
        "label": "Healthcare Utilisation Rate",
        "format": "{:.1f}%",
        "description": "Percentage of population using Medicare services",
        "good_threshold": 80.0,
        "bad_threshold": 60.0,
    },
    "bulk_billed_percentage": {
        "label": "Bulk Billing Rate",
        "format": "{:.1f}%",
        "description": "Percentage of services bulk billed",
        "good_threshold": 80.0,
        "bad_threshold": 60.0,
    },
    "composite_health_index": {
        "label": "Composite Health Index",
        "format": "{:.1f}",
        "description": "Overall health score (higher is better)",
        "good_threshold": 75.0,
        "bad_threshold": 50.0,
    },
    "seifa_irsad_score": {
        "label": "SEIFA IRSAD Score",
        "format": "{:.0f}",
        "description": "Index of Relative Socio-economic Advantage and Disadvantage",
        "good_threshold": 1000,
        "bad_threshold": 900,
    },
}

# Filter options
REMOTENESS_CATEGORIES = [
    "Major Cities of Australia",
    "Inner Regional Australia",
    "Outer Regional Australia",
    "Remote Australia",
    "Very Remote Australia",
]

STATE_CODES = {
    "1": "New South Wales",
    "2": "Victoria",
    "3": "Queensland",
    "4": "South Australia",
    "5": "Western Australia",
    "6": "Tasmania",
    "7": "Northern Territory",
    "8": "Australian Capital Territory",
}

# Page configuration
PAGES = {
    "Overview": {
        "icon": "📊",
        "description": "High-level summary of health indicators across Australia",
    },
    "Geographic Analysis": {
        "icon": "🗺️",
        "description": "Interactive maps and spatial analysis",
    },
    "Health Indicators": {
        "icon": "🏥",
        "description": "Deep dive into health metrics and outcomes",
    },
    "Socioeconomic Impact": {
        "icon": "💰",
        "description": "Correlation between socioeconomic factors and health",
    },
    "Climate & Environment": {
        "icon": "🌡️",
        "description": "Environmental factors affecting health",
    },
    "Data Quality": {
        "icon": "✅",
        "description": "Pipeline status and data quality metrics",
    },
}

# Export configuration
EXPORT_FORMATS = ["CSV", "Excel", "Parquet"]
MAX_EXPORT_ROWS = 100000
