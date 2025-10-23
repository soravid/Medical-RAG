"""
医疗推荐系统 - 提示词模板模块
提供各种场景的提示词模板
"""

from .diagnosis_prompts import DiagnosisPrompts
from .recommendation_prompts import RecommendationPrompts
from .qa_prompts import QAPrompts

__all__ = [
    "DiagnosisPrompts",
    "RecommendationPrompts",
    "QAPrompts",
]