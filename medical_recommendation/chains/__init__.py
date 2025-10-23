"""
医疗推荐系统 - 处理链模块
提供诊断、药品推荐、治疗方案和问答等处理链
"""

from .diagnosis_chain import DiagnosisChain, DiagnosisInput, DiagnosisOutput
from .drug_recommendation_chain import (
    DrugRecommendationChain,
    DrugRecommendationInput,
    DrugRecommendationOutput
)
from .treatment_plan_chain import (
    TreatmentPlanChain,
    TreatmentPlanInput,
    TreatmentPlanOutput
)
from .qa_chain import QAChain, QAInput, QAOutput

__all__ = [
    # 诊断链
    "DiagnosisChain",
    "DiagnosisInput",
    "DiagnosisOutput",
    
    # 药品推荐链
    "DrugRecommendationChain",
    "DrugRecommendationInput",
    "DrugRecommendationOutput",
    
    # 治疗方案链
    "TreatmentPlanChain",
    "TreatmentPlanInput",
    "TreatmentPlanOutput",
    
    # 问答链
    "QAChain",
    "QAInput",
    "QAOutput",
]