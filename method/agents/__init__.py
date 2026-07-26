"""Agent building blocks for the feedback-aware PPT repair workflow."""

from .client import ClientAgent
from .content_validation import ContentValidationAgent
from .data_source_validation import DataSourceValidationAgent
from .slide_analysis import SlideAnalysisAgent
from .slide_parser import SlideParserAgent, extract_pptx_elements
from .types import (
    DetectedIssue,
    SlideReviewInput,
    SlideReviewResult,
)

__all__ = [
    "ClientAgent",
    "ContentValidationAgent",
    "DataSourceValidationAgent",
    "DetectedIssue",
    "SlideReviewInput",
    "SlideReviewResult",
    "SlideAnalysisAgent",
    "SlideParserAgent",
    "extract_pptx_elements",
]
