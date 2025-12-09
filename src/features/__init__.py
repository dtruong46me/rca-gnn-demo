"""
Features package initialization.
"""

from .feature_engineering import (
    fit_static_encoders,
    build_static_feature_matrix,
    compute_degree_features,
    aggregate_events_for_window,
    build_combined_features
)

__all__ = [
    'fit_static_encoders',
    'build_static_feature_matrix',
    'compute_degree_features',
    'aggregate_events_for_window',
    'build_combined_features'
]
