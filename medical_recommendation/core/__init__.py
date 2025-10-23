"""
医疗推荐系统 - 核心引擎模块
整合所有组件提供统一的推荐服务
"""

from .patient_profiling import PatientProfileManager, PatientProfile
from .knowledge_graph import KnowledgeGraphManager
from .vector_store import VectorStoreManager
from .hybrid_retriever import HybridRetriever
from .llm_chain import LLMChainManager
from .recommendation_engine import RecommendationEngine, RecommendationRequest, RecommendationResponse

__all__ = [
    # 患者画像
    "PatientProfileManager",
    "PatientProfile",
    
    # 知识图谱
    "KnowledgeGraphManager",
    
    # 向量存储
    "VectorStoreManager",
    
    # 混合检索
    "HybridRetriever",
    
    # LLM链
    "LLMChainManager",
    
    # 推荐引擎
    "RecommendationEngine",
    "RecommendationRequest",
    "RecommendationResponse",
]