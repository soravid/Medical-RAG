"""
治疗方案链
基于诊断结果生成综合治疗方案
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TreatmentPlanInput(BaseModel):
    """治疗方案链输入模型"""
    patient_id: str = Field(..., description="患者ID")
    diagnosis: str = Field(..., description="诊断结果")
    disease_id: Optional[str] = Field(None, description="疾病ID")
    severity: str = Field(default="moderate", description="疾病严重程度")
    patient_profile: Optional[Dict[str, Any]] = Field(
        None,
        description="患者档案"
    )
    current_treatments: List[str] = Field(
        default_factory=list,
        description="当前治疗措施"
    )
    treatment_goals: List[str] = Field(
        default_factory=list,
        description="治疗目标"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="治疗约束（经济、时间等）"
    )
    preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description="治疗偏好"
    )


class TreatmentPhase(BaseModel):
    """治疗阶段"""
    phase_name: str = Field(..., description="阶段名称")
    duration: str = Field(..., description="持续时间")
    goals: List[str] = Field(default_factory=list, description="阶段目标")
    interventions: List[str] = Field(
        default_factory=list,
        description="治疗措施"
    )
    medications: List[str] = Field(
        default_factory=list,
        description="用药方案"
    )
    monitoring: List[str] = Field(
        default_factory=list,
        description="监测指标"
    )
    expected_outcomes: List[str] = Field(
        default_factory=list,
        description="预期结果"
    )


class LifestyleRecommendation(BaseModel):
    """生活方式建议"""
    category: str = Field(..., description="类别（饮食/运动/作息等）")
    recommendations: List[str] = Field(
        default_factory=list,
        description="具体建议"
    )
    dos: List[str] = Field(default_factory=list, description="应该做的")
    donts: List[str] = Field(default_factory=list, description="不应该做的")
    importance: str = Field(
        default="medium",
        description="重要性（low/medium/high）"
    )


class FollowUpPlan(BaseModel):
    """复诊计划"""
    schedule: str = Field(..., description="复诊时间安排")
    check_items: List[str] = Field(
        default_factory=list,
        description="检查项目"
    )
    evaluation_criteria: List[str] = Field(
        default_factory=list,
        description="评估标准"
    )
    adjustment_triggers: List[str] = Field(
        default_factory=list,
        description="需要调整方案的触发条件"
    )


class TreatmentPlanOutput(BaseModel):
    """治疗方案链输出模型"""
    patient_id: str = Field(..., description="患者ID")
    diagnosis: str = Field(..., description="诊断")
    treatment_strategy: str = Field(..., description="治疗策略概述")
    phases: List[TreatmentPhase] = Field(
        default_factory=list,
        description="治疗阶段"
    )
    medication_plan: Dict[str, Any] = Field(
        default_factory=dict,
        description="用药计划详情"
    )
    lifestyle_recommendations: List[LifestyleRecommendation] = Field(
        default_factory=list,
        description="生活方式建议"
    )
    follow_up_plan: Optional[FollowUpPlan] = Field(
        None,
        description="复诊计划"
    )
    emergency_guidelines: List[str] = Field(
        default_factory=list,
        description="紧急情况处理指南"
    )
    estimated_duration: str = Field(..., description="预计治疗时长")
    expected_outcomes: List[str] = Field(
        default_factory=list,
        description="预期治疗结果"
    )
    risks_and_complications: List[str] = Field(
        default_factory=list,
        description="可能的风险和并发症"
    )
    cost_estimate: Optional[Dict[str, Any]] = Field(
        None,
        description="费用估算"
    )
    retrieved_knowledge: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="检索到的治疗知识"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="元数据"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="生成时间"
    )


class TreatmentPlanChain:
    """
    治疗方案链
    基于诊断结果生成综合治疗方案
    """
    
    def __init__(
        self,
        retriever: Optional[Any] = None,
        llm: Optional[Any] = None,
        use_mock: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化治疗方案链
        
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
        self.include_lifestyle = self.config.get("include_lifestyle", True)
        self.include_cost_estimate = self.config.get("include_cost_estimate", False)
        
        if self.retriever is None and not use_mock:
            logger.warning("未提供检索器，将使用模拟方案生成")
            self.use_mock = True
    
    def run(self, input_data: TreatmentPlanInput) -> TreatmentPlanOutput:
        """
        执行治疗方案链
        
        Args:
            input_data: 输入数据
            
        Returns:
            治疗方案输出
        """
        logger.info(f"开始生成治疗方案: 患者ID={input_data.patient_id}, 诊断={input_data.diagnosis}")
        
        try:
            # 1. 检索治疗知识
            knowledge = self._retrieve_treatment_knowledge(input_data)
            logger.info(f"检索到 {len(knowledge)} 条治疗知识")
            
            # 2. 生成治疗策略
            strategy = self._generate_treatment_strategy(input_data, knowledge)
            
            # 3. 规划治疗阶段
            phases = self._plan_treatment_phases(input_data, strategy)
            logger.info(f"规划 {len(phases)} 个治疗阶段")
            
            # 4. 制定用药计划
            medication_plan = self._create_medication_plan(input_data, phases)
            
            # 5. 生成生活方式建议
            lifestyle_recommendations = []
            if self.include_lifestyle:
                lifestyle_recommendations = self._generate_lifestyle_recommendations(input_data)
            
            # 6. 制定复诊计划
            follow_up_plan = self._create_follow_up_plan(input_data, phases)
            
            # 7. 生成紧急情况指南
            emergency_guidelines = self._generate_emergency_guidelines(input_data)
            
            # 8. 估算治疗时长
            estimated_duration = self._estimate_treatment_duration(phases)
            
            # 9. 预期结果和风险
            expected_outcomes = self._define_expected_outcomes(input_data)
            risks = self._identify_risks_and_complications(input_data)
            
            # 10. 费用估算（可选）
            cost_estimate = None
            if self.include_cost_estimate:
                cost_estimate = self._estimate_costs(input_data, phases)
            
            # 构建输出
            output = TreatmentPlanOutput(
                patient_id=input_data.patient_id,
                diagnosis=input_data.diagnosis,
                treatment_strategy=strategy,
                phases=phases,
                medication_plan=medication_plan,
                lifestyle_recommendations=lifestyle_recommendations,
                follow_up_plan=follow_up_plan,
                emergency_guidelines=emergency_guidelines,
                estimated_duration=estimated_duration,
                expected_outcomes=expected_outcomes,
                risks_and_complications=risks,
                cost_estimate=cost_estimate,
                retrieved_knowledge=knowledge,
                metadata={
                    "severity": input_data.severity,
                    "num_phases": len(phases)
                }
            )
            
            logger.info(f"治疗方案生成完成: {len(phases)}个阶段, 预计{estimated_duration}")
            return output
        
        except Exception as e:
            logger.error(f"治疗方案生成过程出错: {e}")
            raise
    
    def _retrieve_treatment_knowledge(
        self,
        input_data: TreatmentPlanInput
    ) -> List[Dict[str, Any]]:
        """检索治疗知识"""
        if self.use_mock or self.retriever is None:
            return self._mock_retrieve_treatment_knowledge(input_data)
        
        # 实际检索逻辑
        try:
            query_text = f"{input_data.diagnosis}的治疗方案"
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
            logger.error(f"治疗知识检索失败: {e}")
            return []
    
    def _generate_treatment_strategy(
        self,
        input_data: TreatmentPlanInput,
        knowledge: List[Dict[str, Any]]
    ) -> str:
        """生成治疗策略"""
        if self.use_mock or self.llm is None:
            return self._mock_generate_strategy(input_data)
        
        # 实际生成逻辑（使用LLM）
        # TODO: 实现LLM调用
        logger.warning("使用模拟策略生成")
        return self._mock_generate_strategy(input_data)
    
    def _plan_treatment_phases(
        self,
        input_data: TreatmentPlanInput,
        strategy: str
    ) -> List[TreatmentPhase]:
        """规划治疗阶段"""
        if self.use_mock:
            return self._mock_plan_phases(input_data)
        
        # 实际规划逻辑
        # TODO: 实现详细的阶段规划
        logger.warning("使用模拟阶段规划")
        return self._mock_plan_phases(input_data)
    
    def _create_medication_plan(
        self,
        input_data: TreatmentPlanInput,
        phases: List[TreatmentPhase]
    ) -> Dict[str, Any]:
        """制定用药计划"""
        medication_plan = {
            "summary": "根据治疗阶段制定的用药计划",
            "phases": []
        }
        
        for phase in phases:
            phase_plan = {
                "phase": phase.phase_name,
                "medications": phase.medications,
                "duration": phase.duration
            }
            medication_plan["phases"].append(phase_plan)
        
        return medication_plan
    
    def _generate_lifestyle_recommendations(
        self,
        input_data: TreatmentPlanInput
    ) -> List[LifestyleRecommendation]:
        """生成生活方式建议"""
        return self._mock_generate_lifestyle_recommendations(input_data)
    
    def _create_follow_up_plan(
        self,
        input_data: TreatmentPlanInput,
        phases: List[TreatmentPhase]
    ) -> FollowUpPlan:
        """制定复诊计划"""
        return FollowUpPlan(
            schedule="治疗开始后1周、1个月、3个月复诊",
            check_items=[
                "症状改善情况评估",
                "药物疗效和副作用监测",
                "相关检查指标复查"
            ],
            evaluation_criteria=[
                "症状明显缓解",
                "检查指标恢复正常范围",
                "无明显不良反应"
            ],
            adjustment_triggers=[
                "症状无改善或加重",
                "出现严重不良反应",
                "检查指标异常"
            ]
        )
    
    def _generate_emergency_guidelines(
        self,
        input_data: TreatmentPlanInput
    ) -> List[str]:
        """生成紧急情况指南"""
        guidelines = [
            "如出现严重不适或症状突然加重，立即就医",
            "如发生药物过敏反应（皮疹、呼吸困难等），立即停药并就医",
            "保持医生联系方式，必要时及时沟通"
        ]
        
        # 根据疾病严重程度添加特定指南
        if input_data.severity in ["severe", "critical"]:
            guidelines.insert(0, "病情严重，需密切观察，如有异常立即急诊就医")
        
        return guidelines
    
    def _estimate_treatment_duration(self, phases: List[TreatmentPhase]) -> str:
        """估算治疗时长"""
        if not phases:
            return "待评估"
        
        # 简单累加各阶段时长
        total_days = 0
        for phase in phases:
            # 解析时长（简化处理）
            if "周" in phase.duration:
                weeks = int(''.join(filter(str.isdigit, phase.duration)) or 0)
                total_days += weeks * 7
            elif "天" in phase.duration:
                days = int(''.join(filter(str.isdigit, phase.duration)) or 0)
                total_days += days
            elif "月" in phase.duration:
                months = int(''.join(filter(str.isdigit, phase.duration)) or 0)
                total_days += months * 30
        
        if total_days < 30:
            return f"{total_days}天"
        elif total_days < 90:
            return f"{total_days // 7}周"
        else:
            return f"{total_days // 30}个月"
    
    def _define_expected_outcomes(self, input_data: TreatmentPlanInput) -> List[str]:
        """定义预期结果"""
        return [
            "症状明显缓解或消失",
            "相关检查指标恢复正常",
            "生活质量得到改善",
            "防止疾病进展和并发症发生"
        ]
    
    def _identify_risks_and_complications(
        self,
        input_data: TreatmentPlanInput
    ) -> List[str]:
        """识别风险和并发症"""
        return [
            "药物不良反应",
            "治疗效果不佳",
            "疾病复发或进展",
            "并发症发生（具体取决于疾病类型）"
        ]
    
    def _estimate_costs(
        self,
        input_data: TreatmentPlanInput,
        phases: List[TreatmentPhase]
    ) -> Dict[str, Any]:
        """估算治疗费用"""
        return {
            "total_estimate": "根据具体治疗方案而定",
            "medication_cost": "待评估",
            "examination_cost": "待评估",
            "consultation_cost": "待评估",
            "note": "具体费用请咨询医院收费处"
        }
    
    def _mock_retrieve_treatment_knowledge(
        self,
        input_data: TreatmentPlanInput
    ) -> List[Dict[str, Any]]:
        """模拟治疗知识检索"""
        return [
            {
                "content": f"{input_data.diagnosis}的标准治疗方案包括药物治疗和生活方式干预。",
                "score": 0.9,
                "source": "kg"
            },
            {
                "content": "治疗应遵循个体化原则，根据患者具体情况调整。",
                "score": 0.85,
                "source": "vector"
            },
        ]
    
    def _mock_generate_strategy(self, input_data: TreatmentPlanInput) -> str:
        """模拟治疗策略生成"""
        severity_map = {
            "mild": "采用保守治疗为主，结合生活方式调整",
            "moderate": "采用药物治疗结合生活方式干预的综合治疗方案",
            "severe": "采用积极的药物治疗，必要时考虑住院治疗",
            "critical": "立即住院治疗，密切监测病情变化"
        }
        
        base_strategy = severity_map.get(input_data.severity, "采用综合治疗方案")
        return f"针对{input_data.diagnosis}，{base_strategy}。治疗目标是控制症状、改善预后、提高生活质量。"
    
    def _mock_plan_phases(self, input_data: TreatmentPlanInput) -> List[TreatmentPhase]:
        """模拟治疗阶段规划"""
        diagnosis_lower = input_data.diagnosis.lower()
        
        # 急性疾病（如感冒）
        if any(kw in diagnosis_lower for kw in ["感冒", "上呼吸道", "急性"]):
            return [
                TreatmentPhase(
                    phase_name="急性期治疗",
                    duration="3-5天",
                    goals=["缓解症状", "控制感染"],
                    interventions=["休息", "多饮水", "药物治疗"],
                    medications=["退热药", "止咳药"],
                    monitoring=["体温", "症状变化"],
                    expected_outcomes=["发热消退", "症状明显改善"]
                ),
                TreatmentPhase(
                    phase_name="恢复期",
                    duration="3-7天",
                    goals=["完全康复", "增强免疫力"],
                    interventions=["适当休息", "营养补充"],
                    medications=["维生素补充剂（可选）"],
                    monitoring=["整体状态"],
                    expected_outcomes=["完全康复", "恢复正常活动"]
                )
            ]
        
        # 慢性疾病（如高血压、糖尿病）
        elif any(kw in diagnosis_lower for kw in ["高血压", "糖尿病", "慢性"]):
            return [
                TreatmentPhase(
                    phase_name="初始治疗期",
                    duration="4-8周",
                    goals=["控制疾病指标", "评估药物疗效"],
                    interventions=["药物治疗", "生活方式调整", "定期监测"],
                    medications=["根据具体疾病选择合适药物"],
                    monitoring=["血压/血糖", "症状", "药物副作用"],
                    expected_outcomes=["指标初步控制", "无明显不良反应"]
                ),
                TreatmentPhase(
                    phase_name="调整优化期",
                    duration="2-3个月",
                    goals=["优化治疗方案", "达到目标值"],
                    interventions=["调整用药", "强化生活方式管理"],
                    medications=["根据疗效调整剂量或更换药物"],
                    monitoring=["定期复查指标", "并发症筛查"],
                    expected_outcomes=["指标稳定达标", "症状良好控制"]
                ),
                TreatmentPhase(
                    phase_name="维持治疗期",
                    duration="长期",
                    goals=["维持稳定", "预防并发症"],
                    interventions=["坚持用药", "定期复诊", "健康管理"],
                    medications=["维持有效剂量"],
                    monitoring=["定期复查", "并发症监测"],
                    expected_outcomes=["长期稳定", "提高生活质量"]
                )
            ]
        
        # 默认通用方案
        else:
            return [
                TreatmentPhase(
                    phase_name="诊断评估期",
                    duration="1-2周",
                    goals=["明确诊断", "评估病情"],
                    interventions=["完善检查", "初步治疗"],
                    medications=["对症治疗药物"],
                    monitoring=["症状变化", "检查结果"],
                    expected_outcomes=["明确诊断", "初步缓解症状"]
                ),
                TreatmentPhase(
                    phase_name="系统治疗期",
                    duration="4-12周",
                    goals=["系统治疗", "控制病情"],
                    interventions=["规范治疗", "定期评估"],
                    medications=["根据诊断选择药物"],
                    monitoring=["疗效评估", "不良反应"],
                    expected_outcomes=["病情控制", "症状明显改善"]
                )
            ]
    
    def _mock_generate_lifestyle_recommendations(
        self,
        input_data: TreatmentPlanInput
    ) -> List[LifestyleRecommendation]:
        """模拟生活方式建议生成"""
        recommendations = []
        
        # 饮食建议
        recommendations.append(LifestyleRecommendation(
            category="饮食",
            recommendations=[
                "保持均衡饮食，多吃蔬菜水果",
                "控制盐分摄入",
                "避免高脂肪、高糖食物",
                "适量摄入优质蛋白"
            ],
            dos=["清淡饮食", "定时定量", "细嚼慢咽"],
            donts=["暴饮暴食", "过度节食", "大量饮酒"],
            importance="high"
        ))
        
        # 运动建议
        recommendations.append(LifestyleRecommendation(
            category="运动",
            recommendations=[
                "每天进行适量运动，如散步、慢跑",
                "运动时间30-60分钟为宜",
                "循序渐进，避免过度劳累"
            ],
            dos=["有氧运动", "适量活动", "运动前热身"],
            donts=["剧烈运动", "长期久坐", "运动后立即休息"],
            importance="high"
        ))
        
        # 作息建议
        recommendations.append(LifestyleRecommendation(
            category="作息",
            recommendations=[
                "保证充足睡眠，每天7-8小时",
                "规律作息，避免熬夜",
                "保持良好心态，减轻压力"
            ],
            dos=["规律作息", "适当放松", "保持心情愉悦"],
            donts=["熬夜", "过度劳累", "长期焦虑"],
            importance="medium"
        ))
        
        # 疾病特定建议
        diagnosis_lower = input_data.diagnosis.lower()
        if "高血压" in diagnosis_lower or "hypertension" in diagnosis_lower:
            recommendations.append(LifestyleRecommendation(
                category="血压管理",
                recommendations=[
                    "每天测量血压并记录",
                    "限制钠盐摄入（每日<6g）",
                    "戒烟限酒",
                    "保持理想体重"
                ],
                dos=["定时监测", "低盐饮食", "控制体重"],
                donts=["高盐饮食", "吸烟", "过量饮酒"],
                importance="high"
            ))
        
        return recommendations
    
    def validate_input(
        self,
        input_data: TreatmentPlanInput
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