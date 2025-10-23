"""
诊断链
基于症状和患者信息进行疾病诊断
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DiagnosisInput(BaseModel):
    """诊断链输入模型"""
    patient_id: str = Field(..., description="患者ID")
    symptoms: List[str] = Field(..., description="症状列表", min_length=1)
    duration: Optional[str] = Field(None, description="症状持续时间")
    severity: Optional[str] = Field(None, description="症状严重程度")
    patient_history: Optional[Dict[str, Any]] = Field(
        None,
        description="患者病史"
    )
    vital_signs: Optional[Dict[str, Any]] = Field(
        None,
        description="生命体征"
    )
    lab_results: Optional[Dict[str, Any]] = Field(
        None,
        description="实验室检查结果"
    )
    additional_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="其他信息"
    )


class DiagnosisCandidate(BaseModel):
    """诊断候选结果"""
    disease_id: str = Field(..., description="疾病ID")
    disease_name: str = Field(..., description="疾病名称")
    confidence: float = Field(..., description="置信度", ge=0.0, le=1.0)
    matched_symptoms: List[str] = Field(
        default_factory=list,
        description="匹配的症状"
    )
    reasoning: str = Field(..., description="诊断推理过程")
    risk_level: Optional[str] = Field(None, description="风险等级")
    recommended_tests: List[str] = Field(
        default_factory=list,
        description="建议的检查项目"
    )


class DiagnosisOutput(BaseModel):
    """诊断链输出模型"""
    patient_id: str = Field(..., description="患者ID")
    primary_diagnosis: Optional[DiagnosisCandidate] = Field(
        None,
        description="主要诊断"
    )
    differential_diagnoses: List[DiagnosisCandidate] = Field(
        default_factory=list,
        description="鉴别诊断列表"
    )
    overall_assessment: str = Field(..., description="总体评估")
    urgency_level: str = Field(
        default="normal",
        description="紧急程度 (normal/urgent/emergency)"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="建议列表"
    )
    confidence_score: float = Field(
        default=0.0,
        description="整体诊断置信度"
    )
    retrieved_knowledge: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="检索到的知识"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="元数据"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="诊断时间"
    )


class BaseDiagnosisChain(ABC):
    """诊断链抽象基类"""
    
    @abstractmethod
    def run(self, input_data: DiagnosisInput) -> DiagnosisOutput:
        """
        执行诊断链
        
        Args:
            input_data: 诊断输入
            
        Returns:
            诊断输出
        """
        pass
    
    @abstractmethod
    def _retrieve_relevant_knowledge(
        self,
        symptoms: List[str]
    ) -> List[Dict[str, Any]]:
        """检索相关知识"""
        pass
    
    @abstractmethod
    def _generate_candidates(
        self,
        input_data: DiagnosisInput,
        knowledge: List[Dict[str, Any]]
    ) -> List[DiagnosisCandidate]:
        """生成候选诊断"""
        pass
    
    @abstractmethod
    def _rank_candidates(
        self,
        candidates: List[DiagnosisCandidate]
    ) -> List[DiagnosisCandidate]:
        """排序候选诊断"""
        pass


class DiagnosisChain(BaseDiagnosisChain):
    """
    诊断链实现
    基于症状和患者信息进行疾病诊断
    """
    
    def __init__(
        self,
        retriever: Optional[Any] = None,
        llm: Optional[Any] = None,
        use_mock: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化诊断链
        
        Args:
            retriever: 检索器实例
            llm: 大语言模型实例
            use_mock: 是否使用模拟数据
            config: 配置字典
        """
        self.retriever = retriever
        self.llm = llm
        self.use_mock = use_mock
        self.config = config or {}
        
        # 默认配置
        self.max_candidates = self.config.get("max_candidates", 5)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        
        if self.retriever is None and not use_mock:
            logger.warning("未提供检索器，将使用模拟诊断")
            self.use_mock = True
    
    def run(self, input_data: DiagnosisInput) -> DiagnosisOutput:
        """
        执行诊断链
        
        Args:
            input_data: 诊断输入
            
        Returns:
            诊断输出
        """
        logger.info(f"开始诊断: 患者ID={input_data.patient_id}, 症状数={len(input_data.symptoms)}")
        
        try:
            # 1. 检索相关知识
            knowledge = self._retrieve_relevant_knowledge(input_data.symptoms)
            logger.info(f"检索到 {len(knowledge)} 条相关知识")
            
            # 2. 生成候选诊断
            candidates = self._generate_candidates(input_data, knowledge)
            logger.info(f"生成 {len(candidates)} 个候选诊断")
            
            # 3. 排序候选诊断
            ranked_candidates = self._rank_candidates(candidates)
            
            # 4. 选择主要诊断和鉴别诊断
            primary = ranked_candidates[0] if ranked_candidates else None
            differential = ranked_candidates[1:self.max_candidates]
            
            # 5. 生成总体评估
            assessment = self._generate_assessment(input_data, primary, differential)
            
            # 6. 确定紧急程度
            urgency = self._determine_urgency(input_data, primary)
            
            # 7. 生成建议
            recommendations = self._generate_recommendations(input_data, primary, differential)
            
            # 8. 计算整体置信度
            confidence = primary.confidence if primary else 0.0
            
            # 构建输出
            output = DiagnosisOutput(
                patient_id=input_data.patient_id,
                primary_diagnosis=primary,
                differential_diagnoses=differential,
                overall_assessment=assessment,
                urgency_level=urgency,
                recommendations=recommendations,
                confidence_score=confidence,
                retrieved_knowledge=knowledge,
                metadata={
                    "num_symptoms": len(input_data.symptoms),
                    "num_candidates": len(candidates),
                    "processing_time": "simulated"
                }
            )
            
            logger.info(f"诊断完成: 主要诊断={primary.disease_name if primary else 'None'}")
            return output
        
        except Exception as e:
            logger.error(f"诊断过程出错: {e}")
            raise
    
    def _retrieve_relevant_knowledge(
        self,
        symptoms: List[str]
    ) -> List[Dict[str, Any]]:
        """
        检索相关知识
        
        Args:
            symptoms: 症状列表
            
        Returns:
            相关知识列表
        """
        if self.use_mock or self.retriever is None:
            return self._mock_retrieve_knowledge(symptoms)
        
        # 实际检索逻辑
        try:
            query_text = "、".join(symptoms)
            results = self.retriever.retrieve(query_text)
            
            knowledge = []
            for result in results:
                knowledge.append({
                    "content": result.content,
                    "score": result.final_score,
                    "source": result.result_type
                })
            
            return knowledge
        except Exception as e:
            logger.error(f"知识检索失败: {e}")
            return []
    
    def _generate_candidates(
        self,
        input_data: DiagnosisInput,
        knowledge: List[Dict[str, Any]]
    ) -> List[DiagnosisCandidate]:
        """
        生成候选诊断
        
        Args:
            input_data: 输入数据
            knowledge: 相关知识
            
        Returns:
            候选诊断列表
        """
        if self.use_mock or self.llm is None:
            return self._mock_generate_candidates(input_data)
        
        # 实际生成逻辑（使用LLM）
        # TODO: 实现LLM调用
        logger.warning("使用模拟候选生成")
        return self._mock_generate_candidates(input_data)
    
    def _rank_candidates(
        self,
        candidates: List[DiagnosisCandidate]
    ) -> List[DiagnosisCandidate]:
        """
        排序候选诊断
        
        Args:
            candidates: 候选诊断列表
            
        Returns:
            排序后的候选诊断
        """
        # 按置信度降序排序
        return sorted(candidates, key=lambda x: x.confidence, reverse=True)
    
    def _generate_assessment(
        self,
        input_data: DiagnosisInput,
        primary: Optional[DiagnosisCandidate],
        differential: List[DiagnosisCandidate]
    ) -> str:
        """生成总体评估"""
        if primary is None:
            return "基于提供的症状信息，无法做出明确诊断。建议进一步检查。"
        
        assessment = f"根据患者的临床表现，主要考虑{primary.disease_name}（置信度：{primary.confidence:.1%}）。"
        
        if differential:
            diff_names = [d.disease_name for d in differential[:3]]
            assessment += f" 需要排除的疾病包括：{' '.join(diff_names)}。"
        
        return assessment
    
    def _determine_urgency(
        self,
        input_data: DiagnosisInput,
        primary: Optional[DiagnosisCandidate]
    ) -> str:
        """确定紧急程度"""
        # 简单的紧急程度判断逻辑
        if input_data.severity and input_data.severity.lower() in ["severe", "重度", "严重"]:
            return "urgent"
        
        if primary and primary.risk_level in ["high", "critical"]:
            return "emergency"
        
        return "normal"
    
    def _generate_recommendations(
        self,
        input_data: DiagnosisInput,
        primary: Optional[DiagnosisCandidate],
        differential: List[DiagnosisCandidate]
    ) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if primary:
            # 添加检查建议
            if primary.recommended_tests:
                recommendations.extend([
                    f"建议进行{test}检查" for test in primary.recommended_tests
                ])
            
            # 添加基本建议
            recommendations.append(f"建议针对{primary.disease_name}进行进一步评估")
        
        # 添加通用建议
        recommendations.append("建议及时就医，获取专业医疗意见")
        recommendations.append("注意观察症状变化")
        
        return recommendations
    
    def _mock_retrieve_knowledge(self, symptoms: List[str]) -> List[Dict[str, Any]]:
        """模拟知识检索"""
        mock_knowledge = [
            {
                "content": "高血压常见症状包括头痛、头晕、心悸等。",
                "score": 0.9,
                "source": "kg"
            },
            {
                "content": "糖尿病典型症状为多饮、多尿、多食、体重下降。",
                "score": 0.85,
                "source": "vector"
            },
            {
                "content": "感冒通常表现为发热、咳嗽、流涕、咽痛。",
                "score": 0.8,
                "source": "kg"
            },
        ]
        return mock_knowledge
    
    def _mock_generate_candidates(
        self,
        input_data: DiagnosisInput
    ) -> List[DiagnosisCandidate]:
        """模拟候选诊断生成"""
        # 基于症状关键词生成模拟候选
        candidates = []
        
        symptoms_text = " ".join(input_data.symptoms).lower()
        
        # 感冒相关
        if any(kw in symptoms_text for kw in ["发热", "咳嗽", "流涕", "fever", "cough"]):
            candidates.append(DiagnosisCandidate(
                disease_id="disease_003",
                disease_name="急性上呼吸道感染（感冒）",
                confidence=0.85,
                matched_symptoms=[s for s in input_data.symptoms if any(
                    kw in s.lower() for kw in ["发热", "咳嗽", "流涕", "fever"]
                )],
                reasoning="患者表现出典型的上呼吸道感染症状，包括发热、咳嗽等。",
                risk_level="low",
                recommended_tests=["血常规", "C反应蛋白"]
            ))
        
        # 高血压相关
        if any(kw in symptoms_text for kw in ["头痛", "头晕", "心悸", "headache", "dizziness"]):
            candidates.append(DiagnosisCandidate(
                disease_id="disease_001",
                disease_name="高血压",
                confidence=0.75,
                matched_symptoms=[s for s in input_data.symptoms if any(
                    kw in s.lower() for kw in ["头痛", "头晕", "心悸"]
                )],
                reasoning="患者症状提示可能存在血压升高，需要监测血压。",
                risk_level="medium",
                recommended_tests=["血压监测", "心电图", "血脂检查"]
            ))
        
        # 糖尿病相关
        if any(kw in symptoms_text for kw in ["口渴", "多饮", "多尿", "thirst", "urination"]):
            candidates.append(DiagnosisCandidate(
                disease_id="disease_002",
                disease_name="糖尿病",
                confidence=0.70,
                matched_symptoms=[s for s in input_data.symptoms if any(
                    kw in s.lower() for kw in ["口渴", "多饮", "多尿"]
                )],
                reasoning="患者出现典型的糖尿病'三多'症状，需要检查血糖水平。",
                risk_level="medium",
                recommended_tests=["空腹血糖", "糖化血红蛋白", "尿常规"]
            ))
        
        # 如果没有匹配的候选，返回通用候选
        if not candidates:
            candidates.append(DiagnosisCandidate(
                disease_id="unknown",
                disease_name="待确诊疾病",
                confidence=0.50,
                matched_symptoms=input_data.symptoms,
                reasoning="基于当前症状信息，需要进一步检查以明确诊断。",
                risk_level="unknown",
                recommended_tests=["血常规", "生化全套", "影像学检查"]
            ))
        
        return candidates
    
    def validate_input(self, input_data: DiagnosisInput) -> tuple[bool, Optional[str]]:
        """
        验证输入数据
        
        Args:
            input_data: 输入数据
            
        Returns:
            (是否有效, 错误信息)
        """
        if not input_data.symptoms:
            return False, "症状列表不能为空"
        
        if not input_data.patient_id:
            return False, "患者ID不能为空"
        
        return True, None