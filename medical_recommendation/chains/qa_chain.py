"""
问答链
处理医疗相关问题并提供答案
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class QuestionType(str, Enum):
    """问题类型枚举"""
    SYMPTOM_INQUIRY = "symptom_inquiry"          # 症状咨询
    DISEASE_INFO = "disease_info"                # 疾病信息
    DRUG_INFO = "drug_info"                      # 药品信息
    TREATMENT_ADVICE = "treatment_advice"        # 治疗建议
    PREVENTION = "prevention"                    # 预防措施
    GENERAL = "general"                          # 一般问题


class QAInput(BaseModel):
    """问答链输入模型"""
    question: str = Field(..., description="用户问题", min_length=1)
    question_type: Optional[QuestionType] = Field(
        None,
        description="问题类型（可自动识别）"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="上下文信息（患者信息、对话历史等）"
    )
    patient_id: Optional[str] = Field(None, description="患者ID")
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="对话历史"
    )
    max_sources: int = Field(default=5, description="最大检索来源数")
    include_references: bool = Field(
        default=True,
        description="是否包含参考来源"
    )


class AnswerSource(BaseModel):
    """答案来源"""
    source_id: str = Field(..., description="来源ID")
    source_type: str = Field(..., description="来源类型（kg/vector/web）")
    content: str = Field(..., description="来源内容")
    relevance_score: float = Field(..., description="相关性分数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class QAOutput(BaseModel):
    """问答链输出模型"""
    question: str = Field(..., description="原始问题")
    answer: str = Field(..., description="答案")
    question_type: QuestionType = Field(..., description="问题类型")
    confidence: float = Field(
        ...,
        description="答案置信度",
        ge=0.0,
        le=1.0
    )
    sources: List[AnswerSource] = Field(
        default_factory=list,
        description="答案来源"
    )
    related_questions: List[str] = Field(
        default_factory=list,
        description="相关问题推荐"
    )
    disclaimer: str = Field(
        default="本答案仅供参考，具体诊疗请咨询专业医生。",
        description="免责声明"
    )
    follow_up_suggestions: List[str] = Field(
        default_factory=list,
        description="后续建议"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="元数据"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="回答时间"
    )


class QAChain:
    """
    问答链
    处理医疗相关问题并提供专业答案
    """
    
    def __init__(
        self,
        retriever: Optional[Any] = None,
        llm: Optional[Any] = None,
        use_mock: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化问答链
        
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
        self.enable_memory = self.config.get("enable_memory", True)
        self.max_context_length = self.config.get("max_context_length", 2000)
        
        if self.retriever is None and not use_mock:
            logger.warning("未提供检索器，将使用模拟问答")
            self.use_mock = True
    
    def run(self, input_data: QAInput) -> QAOutput:
        """
        执行问答链
        
        Args:
            input_data: 问答输入
            
        Returns:
            问答输出
        """
        logger.info(f"开始处理问题: {input_data.question[:50]}...")
        
        try:
            # 1. 识别问题类型
            question_type = input_data.question_type or self._classify_question(
                input_data.question
            )
            logger.info(f"问题类型: {question_type}")
            
            # 2. 检索相关知识
            sources = self._retrieve_relevant_knowledge(
                input_data.question,
                question_type,
                input_data.max_sources
            )
            logger.info(f"检索到 {len(sources)} 个相关来源")
            
            # 3. 构建上下文
            context = self._build_context(input_data, sources)
            
            # 4. 生成答案
            answer = self._generate_answer(
                input_data.question,
                context,
                question_type
            )
            
            # 5. 计算置信度
            confidence = self._calculate_confidence(sources, answer)
            
            # 6. 生成相关问题
            related_questions = self._generate_related_questions(
                input_data.question,
                question_type
            )
            
            # 7. 生成后续建议
            follow_up_suggestions = self._generate_follow_up_suggestions(
                question_type,
                answer
            )
            
            # 构建输出
            output = QAOutput(
                question=input_data.question,
                answer=answer,
                question_type=question_type,
                confidence=confidence,
                sources=sources if input_data.include_references else [],
                related_questions=related_questions,
                follow_up_suggestions=follow_up_suggestions,
                metadata={
                    "num_sources": len(sources),
                    "context_length": len(context)
                }
            )
            
            logger.info(f"问答完成: 置信度={confidence:.2f}")
            return output
        
        except Exception as e:
            logger.error(f"问答处理过程出错: {e}")
            raise
    
    def _classify_question(self, question: str) -> QuestionType:
        """
        分类问题类型
        
        Args:
            question: 问题文本
            
        Returns:
            问题类型
        """
        question_lower = question.lower()
        
        # 简单的关键词匹配分类
        if any(kw in question_lower for kw in ["症状", "表现", "感觉", "symptom"]):
            return QuestionType.SYMPTOM_INQUIRY
        elif any(kw in question_lower for kw in ["什么病", "疾病", "disease", "诊断"]):
            return QuestionType.DISEASE_INFO
        elif any(kw in question_lower for kw in ["药", "medication", "drug", "用药"]):
            return QuestionType.DRUG_INFO
        elif any(kw in question_lower for kw in ["治疗", "怎么办", "treatment", "cure"]):
            return QuestionType.TREATMENT_ADVICE
        elif any(kw in question_lower for kw in ["预防", "避免", "prevention", "prevent"]):
            return QuestionType.PREVENTION
        else:
            return QuestionType.GENERAL
    
    def _retrieve_relevant_knowledge(
        self,
        question: str,
        question_type: QuestionType,
        max_sources: int
    ) -> List[AnswerSource]:
        """检索相关知识"""
        if self.use_mock or self.retriever is None:
            return self._mock_retrieve_knowledge(question, question_type)
        
        # 实际检索逻辑
        try:
            results = self.retriever.retrieve(question, top_k=max_sources)
            
            sources = []
            for i, result in enumerate(results):
                source = AnswerSource(
                    source_id=f"source_{i+1}",
                    source_type=result.result_type,
                    content=result.content,
                    relevance_score=result.final_score,
                    metadata=result.metadata
                )
                sources.append(source)
            
            return sources
        except Exception as e:
            logger.error(f"知识检索失败: {e}")
            return []
    
    def _build_context(
        self,
        input_data: QAInput,
        sources: List[AnswerSource]
    ) -> str:
        """构建上下文"""
        context_parts = []
        
        # 添加检索到的知识
        for source in sources:
            context_parts.append(f"[来源 {source.source_id}]: {source.content}")
        
        # 添加对话历史（如果启用记忆）
        if self.enable_memory and input_data.conversation_history:
            context_parts.append("\n对话历史:")
            for i, conv in enumerate(input_data.conversation_history[-3:]):  # 只保留最近3轮
                context_parts.append(f"Q: {conv.get('question', '')}")
                context_parts.append(f"A: {conv.get('answer', '')}")
        
        # 添加患者上下文信息
        if input_data.context:
            context_parts.append(f"\n相关信息: {input_data.context}")
        
        context = "\n".join(context_parts)
        
        # 截断过长的上下文
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "..."
        
        return context
    
    def _generate_answer(
        self,
        question: str,
        context: str,
        question_type: QuestionType
    ) -> str:
        """生成答案"""
        if self.use_mock or self.llm is None:
            return self._mock_generate_answer(question, question_type)
        
        # 实际答案生成（使用LLM）
        # TODO: 实现LLM调用
        logger.warning("使用模拟答案生成")
        return self._mock_generate_answer(question, question_type)
    
    def _calculate_confidence(
        self,
        sources: List[AnswerSource],
        answer: str
    ) -> float:
        """计算答案置信度"""
        if not sources:
            return 0.5
        
        # 基于来源数量和相关性分数计算置信度
        avg_relevance = sum(s.relevance_score for s in sources) / len(sources)
        source_factor = min(len(sources) / 5.0, 1.0)  # 来源越多置信度越高
        
        confidence = (avg_relevance * 0.7 + source_factor * 0.3)
        
        return min(max(confidence, 0.0), 1.0)
    
    def _generate_related_questions(
        self,
        question: str,
        question_type: QuestionType
    ) -> List[str]:
        """生成相关问题"""
        related = []
        
        if question_type == QuestionType.SYMPTOM_INQUIRY:
            related = [
                "这种症状严重吗？",
                "需要立即就医吗？",
                "有什么缓解方法？"
            ]
        elif question_type == QuestionType.DISEASE_INFO:
            related = [
                "这个疾病的常见症状是什么？",
                "如何预防这个疾病？",
                "治疗方法有哪些？"
            ]
        elif question_type == QuestionType.DRUG_INFO:
            related = [
                "这个药有什么副作用？",
                "用药注意事项有哪些？",
                "有替代药品吗？"
            ]
        elif question_type == QuestionType.TREATMENT_ADVICE:
            related = [
                "治疗周期大概多长？",
                "治愈率如何？",
                "有什么辅助治疗方法？"
            ]
        elif question_type == QuestionType.PREVENTION:
            related = [
                "日常生活中要注意什么？",
                "饮食上有什么建议？",
                "需要定期检查吗？"
            ]
        else:
            related = [
                "还有其他相关信息吗？",
                "需要注意什么？",
                "如何获取更多帮助？"
            ]
        
        return related
    
    def _generate_follow_up_suggestions(
        self,
        question_type: QuestionType,
        answer: str
    ) -> List[str]:
        """生成后续建议"""
        suggestions = []
        
        # 通用建议
        suggestions.append("如有疑问，建议咨询专业医生")
        
        # 根据问题类型添加特定建议
        if question_type == QuestionType.SYMPTOM_INQUIRY:
            suggestions.append("如症状持续或加重，请及时就医")
            suggestions.append("注意观察症状变化")
        elif question_type == QuestionType.DRUG_INFO:
            suggestions.append("用药前请仔细阅读说明书")
            suggestions.append("如有不良反应，请立即停药并咨询医生")
        elif question_type == QuestionType.TREATMENT_ADVICE:
            suggestions.append("请在医生指导下进行治疗")
            suggestions.append("定期复诊，监测治疗效果")
        
        return suggestions
    
    def _mock_retrieve_knowledge(
        self,
        question: str,
        question_type: QuestionType
    ) -> List[AnswerSource]:
        """模拟知识检索"""
        mock_sources = []
        
        if question_type == QuestionType.SYMPTOM_INQUIRY:
            mock_sources = [
                AnswerSource(
                    source_id="source_1",
                    source_type="kg",
                    content="头痛可能由多种原因引起，包括感冒、高血压、偏头痛等。",
                    relevance_score=0.9,
                    metadata={"category": "symptom"}
                ),
                AnswerSource(
                    source_id="source_2",
                    source_type="vector",
                    content="如果头痛伴随发热、颈部僵硬等症状，需要及时就医。",
                    relevance_score=0.85,
                    metadata={"category": "warning"}
                )
            ]
        elif question_type == QuestionType.DISEASE_INFO:
            mock_sources = [
                AnswerSource(
                    source_id="source_1",
                    source_type="kg",
                    content="高血压是一种慢性疾病，主要表现为血压持续升高。",
                    relevance_score=0.92,
                    metadata={"category": "disease"}
                ),
                AnswerSource(
                    source_id="source_2",
                    source_type="vector",
                    content="高血压需要长期管理，包括药物治疗和生活方式调整。",
                    relevance_score=0.88,
                    metadata={"category": "treatment"}
                )
            ]
        elif question_type == QuestionType.DRUG_INFO:
            mock_sources = [
                AnswerSource(
                    source_id="source_1",
                    source_type="kg",
                    content="阿司匹林是一种常用的解热镇痛药，具有抗炎作用。",
                    relevance_score=0.95,
                    metadata={"category": "drug"}
                ),
                AnswerSource(
                    source_id="source_2",
                    source_type="vector",
                    content="阿司匹林的常见副作用包括胃肠道不适，长期使用需注意。",
                    relevance_score=0.87,
                    metadata={"category": "side_effects"}
                )
            ]
        else:
            mock_sources = [
                AnswerSource(
                    source_id="source_1",
                    source_type="kg",
                    content="相关医疗知识内容。",
                    relevance_score=0.8,
                    metadata={"category": "general"}
                )
            ]
        
        return mock_sources
    
    def _mock_generate_answer(
        self,
        question: str,
        question_type: QuestionType
    ) -> str:
        """模拟答案生成"""
        question_lower = question.lower()
        
        # 症状咨询
        if question_type == QuestionType.SYMPTOM_INQUIRY:
            if "头痛" in question_lower or "headache" in question_lower:
                return ("头痛是一种常见症状，可能由多种原因引起，包括：\n"
                       "1. 感冒或发热\n"
                       "2. 紧张性头痛（压力、疲劳）\n"
                       "3. 偏头痛\n"
                       "4. 高血压\n"
                       "5. 其他疾病\n\n"
                       "如果头痛持续或伴有其他严重症状（如发热、呕吐、意识改变），建议及时就医。"
                       "轻度头痛可以通过休息、放松来缓解。")
            else:
                return "根据您描述的症状，建议进一步观察或就医检查以明确原因。请注意休息，如症状加重请及时就医。"
        
        # 疾病信息
        elif question_type == QuestionType.DISEASE_INFO:
            if "高血压" in question_lower or "hypertension" in question_lower:
                return ("高血压是指动脉血压持续升高的慢性疾病。\n\n"
                       "主要特点：\n"
                       "- 收缩压≥140mmHg 和/或舒张压≥90mmHg\n"
                       "- 常见症状：头痛、头晕、心悸等\n"
                       "- 需要长期管理和治疗\n\n"
                       "治疗方式：\n"
                       "1. 生活方式干预（低盐饮食、适量运动、控制体重）\n"
                       "2. 药物治疗（根据医生处方）\n"
                       "3. 定期监测血压")
            elif "糖尿病" in question_lower or "diabetes" in question_lower:
                return ("糖尿病是一种代谢性疾病，特征是血糖水平持续升高。\n\n"
                       "典型症状（'三多一少'）：\n"
                       "- 多饮、多尿、多食\n"
                       "- 体重下降\n\n"
                       "治疗包括：\n"
                       "1. 饮食控制\n"
                       "2. 适量运动\n"
                       "3. 药物治疗（口服降糖药或胰岛素）\n"
                       "4. 定期监测血糖")
            else:
                return "这是一种需要专业医生诊断和治疗的疾病。建议您咨询医生获取详细信息和个性化治疗方案。"
        
        # 药品信息
        elif question_type == QuestionType.DRUG_INFO:
            if "阿司匹林" in question_lower or "aspirin" in question_lower:
                return ("阿司匹林（乙酰水杨酸）是一种常用药物。\n\n"
                       "主要作用：\n"
                       "- 解热镇痛\n"
                       "- 抗炎\n"
                       "- 抗血小板（小剂量）\n\n"
                       "常见用途：\n"
                       "- 缓解轻至中度疼痛、发热\n"
                       "- 预防心血管疾病（在医生指导下）\n\n"
                       "注意事项：\n"
                       "- 可能引起胃肠道不适\n"
                       "- 有出血倾向者慎用\n"
                       "- 儿童病毒感染发热时不宜使用")
            else:
                return "关于该药品的具体信息，建议您查看药品说明书或咨询药师、医生。请在专业指导下使用药物。"
        
        # 治疗建议
        elif question_type == QuestionType.TREATMENT_ADVICE:
            return ("治疗建议需要根据具体诊断和病情制定。一般原则包括：\n"
                   "1. 在医生指导下进行规范治疗\n"
                   "2. 按时按量用药\n"
                   "3. 定期复诊和检查\n"
                   "4. 注意生活方式调整\n"
                   "5. 监测病情变化\n\n"
                   "建议您前往医院就诊，获取专业的诊疗方案。")
        
        # 预防措施
        elif question_type == QuestionType.PREVENTION:
            return ("预防疾病的一般建议：\n"
                   "1. 保持健康的生活方式\n"
                   "2. 均衡饮食，适量运动\n"
                   "3. 充足睡眠，减轻压力\n"
                   "4. 定期体检\n"
                   "5. 避免不良习惯（吸烟、酗酒等）\n"
                   "6. 注意个人卫生\n"
                   "7. 及时接种疫苗\n\n"
                   "具体预防措施请根据疾病类型咨询医生。")
        
        # 一般问题
        else:
            return ("感谢您的提问。关于这个问题，建议您：\n"
                   "1. 咨询专业医生获取准确信息\n"
                   "2. 如需紧急帮助，请拨打急救电话或前往医院\n"
                   "3. 关注权威医疗健康网站获取可靠信息\n\n"
                   "本系统提供的信息仅供参考，不能替代专业医疗建议。")
    
    def validate_input(self, input_data: QAInput) -> tuple[bool, Optional[str]]:
        """
        验证输入数据
        
        Args:
            input_data: 输入数据
            
        Returns:
            (是否有效, 错误信息)
        """
        if not input_data.question or not input_data.question.strip():
            return False, "问题不能为空"
        
        if len(input_data.question) > 500:
            return False, "问题长度不能超过500字符"
        
        return True, None
    
    def add_to_history(
        self,
        conversation_history: List[Dict[str, str]],
        question: str,
        answer: str
    ) -> List[Dict[str, str]]:
        """
        添加对话到历史记录
        
        Args:
            conversation_history: 现有对话历史
            question: 问题
            answer: 答案
            
        Returns:
            更新后的对话历史
        """
        conversation_history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
        
        # 只保留最近的10轮对话
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        
        return conversation_history