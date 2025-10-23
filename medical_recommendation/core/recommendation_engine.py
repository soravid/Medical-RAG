"""
推荐引擎主逻辑
整合所有模块提供统一的推荐服务
"""

import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from .patient_profiling import PatientProfileManager
from .knowledge_graph import KnowledgeGraphManager
from .vector_store import VectorStoreManager
from .hybrid_retriever import HybridRetriever
from .llm_chain import LLMChainManager
from ..agents import MedicalAgent, AgentInput, AgentMode
from ..chains import (
    DiagnosisChain,
    DrugRecommendationChain,
    TreatmentPlanChain,
    QAChain
)

logger = logging.getLogger(__name__)


class RecommendationRequest(BaseModel):
    """推荐请求模型"""
    request_type: str = Field(
        ...,
        description="请求类型 (diagnosis/drug/treatment/qa/consultation)"
    )
    patient_id: Optional[str] = Field(None, description="患者ID")
    query: str = Field(..., description="查询内容")
    context: Dict[str, Any] = Field(default_factory=dict, description="上下文信息")
    options: Dict[str, Any] = Field(default_factory=dict, description="选项设置")


class RecommendationResponse(BaseModel):
    """推荐响应模型"""
    success: bool = Field(..., description="是否成功")
    request_type: str = Field(..., description="请求类型")
    result: Dict[str, Any] = Field(..., description="结果数据")
    confidence: float = Field(default=0.0, description="置信度")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="信息来源")
    recommendations: List[str] = Field(default_factory=list, description="建议")
    warnings: List[str] = Field(default_factory=list, description="警告")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    error_message: Optional[str] = Field(None, description="错误信息")


class RecommendationEngine:
    """
    推荐引擎
    系统的核心组件，整合所有模块提供统一的推荐服务
    """
    
    def __init__(
        self,
        use_mock: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化推荐引擎
        
        Args:
            use_mock: 是否使用模拟数据
            config: 配置字典
        """
        self.use_mock = use_mock
        self.config = config or {}
        
        logger.info("开始初始化推荐引擎...")
        
        # 初始化核心组件
        self.patient_manager = PatientProfileManager(use_mock=use_mock)
        self.kg_manager = KnowledgeGraphManager(use_mock=use_mock)
        self.vector_store = VectorStoreManager(use_mock=use_mock)
        self.hybrid_retriever = HybridRetriever(
            kg_manager=self.kg_manager,
            vector_store=self.vector_store,
            use_mock=use_mock,
            config=self.config.get("retriever", {})
        )
        self.llm_manager = LLMChainManager(
            use_mock=use_mock,
            config=self.config.get("llm", {})
        )
        
        # 初始化处理链
        self.diagnosis_chain = DiagnosisChain(
            retriever=self.hybrid_retriever,
            use_mock=use_mock
        )
        self.drug_chain = DrugRecommendationChain(
            retriever=self.hybrid_retriever,
            use_mock=use_mock
        )
        self.treatment_chain = TreatmentPlanChain(
            retriever=self.hybrid_retriever,
            use_mock=use_mock
        )
        self.qa_chain = QAChain(
            retriever=self.hybrid_retriever,
            use_mock=use_mock
        )
        
        # 初始化Agent
        self.agent = MedicalAgent(
            use_mock=use_mock,
            config=self.config.get("agent", {})
        )
        
        logger.info("推荐引擎初始化完成")
    
    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        """
        执行推荐
        
        Args:
            request: 推荐请求
            
        Returns:
            推荐响应
        """
        logger.info(f"收到推荐请求: type={request.request_type}, patient={request.patient_id}")
        
        try:
            # 根据请求类型分发处理
            if request.request_type == "diagnosis":
                return self._handle_diagnosis(request)
            elif request.request_type == "drug":
                return self._handle_drug_recommendation(request)
            elif request.request_type == "treatment":
                return self._handle_treatment_plan(request)
            elif request.request_type == "qa":
                return self._handle_qa(request)
            elif request.request_type == "consultation":
                return self._handle_consultation(request)
            else:
                raise ValueError(f"不支持的请求类型: {request.request_type}")
        
        except Exception as e:
            logger.error(f"推荐处理失败: {e}")
            return RecommendationResponse(
                success=False,
                request_type=request.request_type,
                result={},
                error_message=str(e)
            )
    
    def _handle_diagnosis(self, request: RecommendationRequest) -> RecommendationResponse:
        """处理诊断请求"""
        from ..chains import DiagnosisInput
        
        # 获取患者画像
        patient_profile = None
        if request.patient_id:
            patient_profile = self.patient_manager.get_profile(request.patient_id)
        
        # 构建诊断输入
        diagnosis_input = DiagnosisInput(
            patient_id=request.patient_id or "unknown",
            symptoms=request.context.get("symptoms", []),
            duration=request.context.get("duration"),
            severity=request.context.get("severity"),
            patient_history=patient_profile.medical_history if patient_profile else None
        )
        
        # 执行诊断链
        diagnosis_output = self.diagnosis_chain.run(diagnosis_input)
        
        # 构建响应
        result = {
            "primary_diagnosis": diagnosis_output.primary_diagnosis.model_dump() if diagnosis_output.primary_diagnosis else None,
            "differential_diagnoses": [d.model_dump() for d in diagnosis_output.differential_diagnoses],
            "assessment": diagnosis_output.overall_assessment,
            "urgency": diagnosis_output.urgency_level
        }
        
        return RecommendationResponse(
            success=True,
            request_type="diagnosis",
            result=result,
            confidence=diagnosis_output.confidence_score,
            sources=[{"content": k["content"], "score": k.get("score", 0)} for k in diagnosis_output.retrieved_knowledge],
            recommendations=diagnosis_output.recommendations,
            warnings=[],
            metadata=diagnosis_output.metadata
        )
    
    def _handle_drug_recommendation(self, request: RecommendationRequest) -> RecommendationResponse:
        """处理药品推荐请求"""
        from ..chains import DrugRecommendationInput
        
        # 获取患者画像
        patient_profile = None
        if request.patient_id:
            patient_profile = self.patient_manager.get_profile(request.patient_id)
        
        # 构建药品推荐输入
        drug_input = DrugRecommendationInput(
            patient_id=request.patient_id or "unknown",
            diagnosis=request.context.get("diagnosis", request.query),
            symptoms=request.context.get("symptoms", []),
            allergies=patient_profile.allergies if patient_profile else [],
            current_medications=patient_profile.current_medications if patient_profile else []
        )
        
        # 执行药品推荐链
        drug_output = self.drug_chain.run(drug_input)
        
        # 构建响应
        result = {
            "primary_drugs": [d.model_dump() for d in drug_output.primary_drugs],
            "adjuvant_drugs": [d.model_dump() for d in drug_output.adjuvant_drugs],
            "otc_recommendations": [d.model_dump() for d in drug_output.otc_recommendations],
            "interaction_alerts": drug_output.interaction_alerts
        }
        
        return RecommendationResponse(
            success=True,
            request_type="drug",
            result=result,
            confidence=drug_output.overall_safety_score,
            sources=[],
            recommendations=drug_output.recommendations,
            warnings=drug_output.warnings,
            metadata=drug_output.metadata
        )
    
    def _handle_treatment_plan(self, request: RecommendationRequest) -> RecommendationResponse:
        """处理治疗方案请求"""
        from ..chains import TreatmentPlanInput
        
        # 获取患者画像
        patient_profile = None
        if request.patient_id:
            patient_profile = self.patient_manager.get_profile(request.patient_id)
        
        # 构建治疗方案输入
        treatment_input = TreatmentPlanInput(
            patient_id=request.patient_id or "unknown",
            diagnosis=request.context.get("diagnosis", request.query),
            severity=request.context.get("severity", "moderate")
        )
        
        # 执行治疗方案链
        treatment_output = self.treatment_chain.run(treatment_input)
        
        # 构建响应
        result = {
            "strategy": treatment_output.treatment_strategy,
            "phases": [p.model_dump() for p in treatment_output.phases],
            "lifestyle_recommendations": [r.model_dump() for r in treatment_output.lifestyle_recommendations],
            "follow_up_plan": treatment_output.follow_up_plan.model_dump() if treatment_output.follow_up_plan else None,
            "duration": treatment_output.estimated_duration
        }
        
        return RecommendationResponse(
            success=True,
            request_type="treatment",
            result=result,
            confidence=0.8,
            sources=[],
            recommendations=treatment_output.emergency_guidelines,
            warnings=treatment_output.risks_and_complications,
            metadata=treatment_output.metadata
        )
    
    def _handle_qa(self, request: RecommendationRequest) -> RecommendationResponse:
        """处理问答请求"""
        from ..chains import QAInput
        
        # 构建问答输入
        qa_input = QAInput(
            question=request.query,
            patient_id=request.patient_id,
            context=request.context
        )
        
        # 执行问答链
        qa_output = self.qa_chain.run(qa_input)
        
        # 构建响应
        result = {
            "question": qa_output.question,
            "answer": qa_output.answer,
            "question_type": qa_output.question_type,
            "related_questions": qa_output.related_questions
        }
        
        return RecommendationResponse(
            success=True,
            request_type="qa",
            result=result,
            confidence=qa_output.confidence,
            sources=[s.model_dump() for s in qa_output.sources],
            recommendations=qa_output.follow_up_suggestions,
            warnings=[],
            metadata=qa_output.metadata
        )
    
    def _handle_consultation(self, request: RecommendationRequest) -> RecommendationResponse:
        """处理咨询请求（使用Agent）"""
        # 构建Agent输入
        agent_input = AgentInput(
            query=request.query,
            mode=AgentMode.CONSULTATION,
            patient_id=request.patient_id,
            context=request.context
        )
        
        # 执行Agent
        agent_output = self.agent.run(agent_input)
        
        # 构建响应
        result = {
            "answer": agent_output.answer,
            "actions": [a.model_dump() for a in agent_output.actions],
            "tools_used": agent_output.tools_used
        }
        
        return RecommendationResponse(
            success=agent_output.success,
            request_type="consultation",
            result=result,
            confidence=agent_output.confidence,
            sources=agent_output.sources,
            recommendations=agent_output.recommendations,
            warnings=agent_output.warnings,
            metadata=agent_output.metadata
        )
    
    def get_patient_profile(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        获取患者画像
        
        Args:
            patient_id: 患者ID
            
        Returns:
            患者画像字典
        """
        profile = self.patient_manager.get_profile(patient_id)
        if profile:
            return profile.model_dump()
        return None
    
    def search_knowledge(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        搜索医疗知识
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            mode: 检索模式
            
        Returns:
            搜索结果
        """
        return self.hybrid_retriever.retrieve(
            query=query,
            top_k=top_k,
            mode=mode
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取系统统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "knowledge_graph": self.kg_manager.get_statistics(),
            "vector_store": self.vector_store.get_statistics(),
            "patient_profiles": len(self.patient_manager._cache),
            "system_status": "operational"
        }
    
    def health_check(self) -> Dict[str, bool]:
        """
        系统健康检查
        
        Returns:
            健康状态字典
        """
        return {
            "patient_manager": True,
            "kg_manager": True,
            "vector_store": True,
            "hybrid_retriever": True,
            "llm_manager": True,
            "diagnosis_chain": True,
            "drug_chain": True,
            "treatment_chain": True,
            "qa_chain": True,
            "agent": True
        }