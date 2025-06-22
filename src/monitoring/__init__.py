"""
Dataset Monitoring and Analytics Package
=======================================

Provides comprehensive monitoring, analytics, and feedback collection
for the AHGD dataset deployed on Hugging Face Hub.

This package includes:
- Usage analytics and tracking
- Data quality monitoring
- User feedback collection
- Performance monitoring
- Automated alerting
"""

from .analytics import (
    DatasetAnalytics, 
    FeedbackCollector, 
    UsageEvent, 
    QualityMetric, 
    UserFeedback,
    create_monitoring_system
)

__all__ = [
    "DatasetAnalytics",
    "FeedbackCollector", 
    "UsageEvent",
    "QualityMetric",
    "UserFeedback",
    "create_monitoring_system"
]

__version__ = "1.0.0"