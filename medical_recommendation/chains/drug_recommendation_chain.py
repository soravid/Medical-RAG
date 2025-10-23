"""
药品推荐链
基于诊断结果和患者信息推荐药品
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

logger = logging.getLogger(__name__)


class DrugRecommendationInput(BaseModel):
    """药品推荐链输入模型"""
    patient_id: str = Field(..., description="患者ID")
    diagnosis: str = Field(..., description="诊断结果")
    disease_id: Optional[str] = Field(None, description="疾病ID")
    symptoms: List[str] = Field(default_factory=list, description="症状列表")
    patient_profile: Optional[Dict[str, Any]] = Field(
        None,
        description="患者档案"
    )
    allergies: List[str] = Field(
        default_factory=list,
        description="药物过敏史"
    )
    current_medications: List[str] = Field(
        default_factory=list,
        description="当前用药"
    )
    contraindications: List[str] = Field(
        default_factory=list,
        description="禁忌症"
    )
    preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description="偏好设置（如口服/注射等）"
    )


class DrugRecommendation(BaseModel):
    """药品推荐结果"""
    drug_id: str = Field(..., description="药品ID")
    drug_name: str = Field(..., description="药品名称")
    generic_name: Optional[str] = Field(None, description="通用名称")
    dosage: str = Field(..., description="推荐剂量")
    frequency: str = Field(..., description="用药频率")
    duration: Optional[str] = Field(None, description="疗程")
    route: str = Field(default="oral", description="给药途径")
    confidence: float = Field(..., description="推荐置信度", ge=0.0, le=1.0)
    reasoning: str = Field(..., description="推荐理由")
    priority: int = Field(default=1, description="优先级（1为最高）")
    interactions: List[str] = Field(
        default_factory=list,
        description="可能的药物相互作用"
    )
    side_effects: List[str] = Field(
        default_factory=list,
        description="常见副作用"
    )
    precautions: List[str] = Field(
        default_factory=list,
        description="注意事项"
    )
    alternatives: List[str] = Field(
        default_factory=list,
        description="替代药品"
    )


class DrugRecommendationOutput(BaseModel):
    """药品推荐链输出模型"""
    patient_id: str = Field(..., description="患者ID")
    primary_drugs: List[DrugRecommendation] = Field(
        default_factory=list,
        description="主要推荐药品"
    )
    adjuvant_drugs: List[DrugRecommendation] = Field(
        default_factory=list,
        description="辅助药品"
    )
    otc_recommendations: List[DrugRecommendation] = Field(
        default_factory=list,
        description="非处方药推荐"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="警告信息"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="用药建议"
    )
    interaction_alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="药物相互作用警报"
    )
    overall_safety_score: float = Field(
        default=1.0,
        description="整体安全性评分",
        ge=0.0,
        le=1.0
    )
    retrieved_knowledge: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="检索到的药品知识"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="元数据"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="推荐时间"
    )


class DrugRecommendationChain:
    """
    药品推荐链
    基于诊断结果和患者信息推荐合适的药品
    """
    
    def __init__(
        self,
        retriever: Optional[Any] = None,
        llm: Optional[Any] = None,
        use_mock: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化药品推荐链
        
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
        self.max_primary_drugs = self.config.get("max_primary_drugs", 3)
        self.max_adjuvant_drugs = self.config.get("max_adjuvant_drugs", 3)
        self.safety_threshold = self.config.get("safety_threshold", 0.7)
        
        if self.retriever is None and not use_mock:
            logger.warning("未提供检索器，将使用模拟推荐")
            self.use_mock = True
    
    def run(self, input_data: DrugRecommendationInput) -> DrugRecommendationOutput:
        """
        执行药品推荐链
        
        Args:
            input_data: 推荐输入
            
        Returns:
            推荐输出
        """
        logger.info(f"开始药品推荐: 患者ID={input_data.patient_id}, 诊断={input_data.diagnosis}")
        
        try:
            # 1. 检索相关药品知识
            knowledge = self._retrieve_drug_knowledge(input_data)
            logger.info(f"检索到 {len(knowledge)} 条药品知识")
            
            # 2. 生成候选药品
            candidates = self._generate_drug_candidates(input_data, knowledge)
            logger.info(f"生成 {len(candidates)} 个候选药品")
            
            # 3. 检查药物过敏和禁忌
            filtered_candidates = self._filter_by_allergies_and_contraindications(
                candidates,
                input_data.allergies,
                input_data.contraindications
            )
            logger.info(f"过滤后剩余 {len(filtered_candidates)} 个候选药品")
            
            # 4. 检查药物相互作用
            interaction_alerts = self._check_drug_interactions(
                filtered_candidates,
                input_data.current_medications
            )
            
            # 5. 排序和分类药品
            primary_drugs, adjuvant_drugs, otc_drugs = self._categorize_drugs(
                filtered_candidates
            )
            
            # 6. 生成警告和建议
            warnings = self._generate_warnings(input_data, primary_drugs, interaction_alerts)
            recommendations = self._generate_recommendations(input_data, primary_drugs)
            
            # 7. 计算安全性评分
            safety_score = self._calculate_safety_score(
                input_data,
                primary_drugs,
                interaction_alerts
            )
            
            # 构建输出
            output = DrugRecommendationOutput(
                patient_id=input_data.patient_id,
                primary_drugs=primary_drugs[:self.max_primary_drugs],
                adjuvant_drugs=adjuvant_drugs[:self.max_adjuvant_drugs],
                otc_recommendations=otc_drugs,
                warnings=warnings,
                recommendations=recommendations,
                interaction_alerts=interaction_alerts,
                overall_safety_score=safety_score,
                retrieved_knowledge=knowledge,
                metadata={
                    "num_candidates": len(candidates),
                    "num_filtered": len(filtered_candidates),
                    "diagnosis": input_data.diagnosis
                }
            )
            
            logger.info(f"推荐完成: 主要药品={len(output.primary_drugs)}, 辅助药品={len(output.adjuvant_drugs)}")
            return output
        
        except Exception as e:
            logger.error(f"药品推荐过程出错: {e}")
            raise
    
    def _retrieve_drug_knowledge(
        self,
        input_data: DrugRecommendationInput
    ) -> List[Dict[str, Any]]:
        """检索药品知识"""
        if self.use_mock or self.retriever is None:
            return self._mock_retrieve_drug_knowledge(input_data)
        
        # 实际检索逻辑
        try:
            query_text = f"治疗{input_data.diagnosis}的药品"
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
            logger.error(f"药品知识检索失败: {e}")
            return []
    
    def _generate_drug_candidates(
        self,
        input_data: DrugRecommendationInput,
        knowledge: List[Dict[str, Any]]
    ) -> List[DrugRecommendation]:
        """生成候选药品"""
        if self.use_mock or self.llm is None:
            return self._mock_generate_candidates(input_data)
        
        # 实际生成逻辑（使用LLM）
        # TODO: 实现LLM调用
        logger.warning("使用模拟候选生成")
        return self._mock_generate_candidates(input_data)
    
    def _filter_by_allergies_and_contraindications(
        self,
        candidates: List[DrugRecommendation],
        allergies: List[str],
        contraindications: List[str]
    ) -> List[DrugRecommendation]:
        """根据过敏史和禁忌症过滤药品"""
        if not allergies and not contraindications:
            return candidates
        
        filtered = []
        allergies_lower = [a.lower() for a in allergies]
        contraindications_lower = [c.lower() for c in contraindications]
        
        for drug in candidates:
            # 检查过敏
            if any(allergy in drug.drug_name.lower() for allergy in allergies_lower):
                logger.warning(f"药品 {drug.drug_name} 可能导致过敏，已排除")
                continue
            
            # 检查禁忌
            has_contraindication = False
            for contra in contraindications_lower:
                if contra in " ".join(drug.precautions).lower():
                    has_contraindication = True
                    break
            
            if has_contraindication:
                logger.warning(f"药品 {drug.drug_name} 存在禁忌症，已排除")
                continue
            
            filtered.append(drug)
        
        return filtered
    
    def _check_drug_interactions(
        self,
        candidates: List[DrugRecommendation],
        current_medications: List[str]
    ) -> List[Dict[str, Any]]:
        """检查药物相互作用"""
        if not current_medications:
            return []
        
        alerts = []
        
        for drug in candidates:
            for current_med in current_medications:
                # 简单的交互检查（实际应查询药物相互作用数据库）
                if current_med.lower() in " ".join(drug.interactions).lower():
                    alerts.append({
                        "drug": drug.drug_name,
                        "interacts_with": current_med,
                        "severity": "moderate",
                        "description": f"{drug.drug_name}可能与{current_med}存在相互作用"
                    })
        
        return alerts
    
    def _categorize_drugs(
        self,
        candidates: List[DrugRecommendation]
    ) -> tuple[List[DrugRecommendation], List[DrugRecommendation], List[DrugRecommendation]]:
        """将药品分类为主要药品、辅助药品和非处方药"""
        primary = []
        adjuvant = []
        otc = []
        
        # 按优先级和置信度排序
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (x.priority, -x.confidence)
        )
        
        for drug in sorted_candidates:
            if drug.priority == 1:
                primary.append(drug)
            elif drug.priority == 2:
                adjuvant.append(drug)
            else:
                otc.append(drug)
        
        return primary, adjuvant, otc
    
    def _generate_warnings(
        self,
        input_data: DrugRecommendationInput,
        primary_drugs: List[DrugRecommendation],
        interaction_alerts: List[Dict[str, Any]]
    ) -> List[str]:
        """生成警告信息"""
        warnings = []
        
        # 过敏警告
        if input_data.allergies:
            warnings.append(f"患者对以下物质过敏: {', '.join(input_data.allergies)}，请注意避免")
        
        # 相互作用警告
        if interaction_alerts:
            warnings.append(f"检测到 {len(interaction_alerts)} 个潜在的药物相互作用")
        
        # 高风险药品警告
        for drug in primary_drugs:
            if "高风险" in drug.reasoning or "严重" in " ".join(drug.side_effects):
                warnings.append(f"{drug.drug_name}可能存在严重副作用，需密切监测")
        
        return warnings
    
    def _generate_recommendations(
        self,
        input_data: DrugRecommendationInput,
        primary_drugs: List[DrugRecommendation]
    ) -> List[str]:
        """生成用药建议"""
        recommendations = [
            "请严格按照医嘱用药，不要自行调整剂量",
            "定期复诊，监测治疗效果和不良反应",
            "如出现严重不适，请立即就医"
        ]
        
        # 针对特定药品的建议
        for drug in primary_drugs:
            if "饭" in drug.frequency or "餐" in drug.frequency:
                recommendations.append(f"{drug.drug_name}需要{drug.frequency}服用")
        
        return recommendations
    
    def _calculate_safety_score(
        self,
        input_data: DrugRecommendationInput,
        primary_drugs: List[DrugRecommendation],
        interaction_alerts: List[Dict[str, Any]]
    ) -> float:
        """计算安全性评分"""
        score = 1.0
        
        # 根据相互作用数量降低分数
        if interaction_alerts:
            score -= len(interaction_alerts) * 0.1
        
        # 根据过敏史降低分数
        if input_data.allergies:
            score -= len(input_data.allergies) * 0.05
        
        # 根据药品副作用降低分数
        total_side_effects = sum(len(drug.side_effects) for drug in primary_drugs)
        score -= total_side_effects * 0.02
        
        return max(0.0, min(1.0, score))
    
    def _mock_retrieve_drug_knowledge(
        self,
        input_data: DrugRecommendationInput
    ) -> List[Dict[str, Any]]:
        """模拟药品知识检索"""
        return [
            {
                "content": "阿司匹林用于解热镇痛，也可用于预防心血管疾病。",
                "score": 0.9,
                "source": "kg"
            },
            {
                "content": "二甲双胍是治疗2型糖尿病的一线用药。",
                "score": 0.85,
                "source": "vector"
            },
        ]
    
    def _mock_generate_candidates(
        self,
        input_data: DrugRecommendationInput
    ) -> List[DrugRecommendation]:
        """模拟候选药品生成"""
        candidates = []
        
        diagnosis_lower = input_data.diagnosis.lower()
        
        # 感冒相关药品
        if any(kw in diagnosis_lower for kw in ["感冒", "上呼吸道", "cold"]):
            candidates.extend([
                DrugRecommendation(
                    drug_id="drug_001",
                    drug_name="对乙酰氨基酚片",
                    generic_name="对乙酰氨基酚",
                    dosage="500mg",
                    frequency="每次1片，每日3次",
                    duration="3-5天",
                    route="口服",
                    confidence=0.9,
                    reasoning="用于缓解发热和轻度疼痛症状",
                    priority=1,
                    side_effects=["偶见恶心", "皮疹"],
                    precautions=["不宜长期或大量使用", "肝肾功能不全者慎用"],
                    alternatives=["布洛芬"]
                ),
                DrugRecommendation(
                    drug_id="drug_005",
                    drug_name="维生素C片",
                    dosage="100mg",
                    frequency="每日2次",
                    duration="1周",
                    route="口服",
                    confidence=0.7,
                    reasoning="辅助增强免疫力",
                    priority=2,
                    side_effects=["大剂量可能引起腹泻"],
                    precautions=["不宜过量服用"]
                )
            ])
        
        # 高血压相关药品
        if any(kw in diagnosis_lower for kw in ["高血压", "hypertension"]):
            candidates.append(
                DrugRecommendation(
                    drug_id="drug_003",
                    drug_name="氨氯地平片",
                    generic_name="氨氯地平",
                    dosage="5mg",
                    frequency="每日1次",
                    duration="长期",
                    route="口服",
                    confidence=0.95,
                    reasoning="钙通道阻滞剂，用于降低血压",
                    priority=1,
                    side_effects=["头痛", "踝部水肿", "面部潮红"],
                    precautions=["需定期监测血压", "不可突然停药"],
                    alternatives=["缬沙坦", "依那普利"]
                )
            )
        
        # 糖尿病相关药品
        if any(kw in diagnosis_lower for kw in ["糖尿病", "diabetes"]):
            candidates.append(
                DrugRecommendation(
                    drug_id="drug_002",
                    drug_name="二甲双胍片",
                    generic_name="盐酸二甲双胍",
                    dosage="500mg",
                    frequency="每日2-3次，餐后服用",
                    duration="长期",
                    route="口服",
                    confidence=0.92,
                    reasoning="双胍类降糖药，改善胰岛素抵抗",
                    priority=1,
                    interactions=["避免与造影剂同时使用"],
                    side_effects=["胃肠道不适", "腹泻", "乳酸酸中毒（罕见）"],
                    precautions=["肾功能不全者慎用", "定期监测血糖和肾功能"],
                    alternatives=["格列齐特", "阿卡波糖"]
                )
            )
        
        # 如果没有特定匹配，返回通用建议
        if not candidates:
            candidates.append(
                DrugRecommendation(
                    drug_id="drug_general",
                    drug_name="待医生开具处方药",
                    dosage="遵医嘱",
                    frequency="遵医嘱",
                    route="遵医嘱",
                    confidence=0.5,
                    reasoning="需要医生根据具体情况开具处方",
                    priority=1,
                    precautions=["请及时就医获取专业诊疗"]
                )
            )
        
        return candidates
    
    def validate_input(
        self,
        input_data: DrugRecommendationInput
    ) -> tuple[bool, Optional[str]]:
        """
        验证输入数据
        
        Args:
            input_data: 输入数据
            
        Returns:
            (是否有效, 错误信息)
        """
        if not input_data.diagnosis:
            return False, "诊断结果不能为空"
        
        if not input_data.patient_id:
            return False, "患者ID不能为空"
        
        return True, None