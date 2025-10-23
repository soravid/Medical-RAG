"""
医疗顾问Agent
整合多种工具提供智能医疗咨询服务
"""

import logging
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from .tools import KGQueryTool, VectorSearchTool, PatientInfoTool

logger = logging.getLogger(__name__)


class AgentMode(str, Enum):
    """Agent模式枚举"""
    CONSULTATION = "consultation"    # 咨询模式
    DIAGNOSIS = "diagnosis"          # 诊断模式
    TREATMENT = "treatment"          # 治疗模式
    QA = "qa"                        # 问答模式


class AgentInput(BaseModel):
    """Agent输入模型"""
    query: str = Field(..., description="用户查询", min_length=1)
    mode: AgentMode = Field(default=AgentMode.CONSULTATION, description="运行模式")
    patient_id: Optional[str] = Field(None, description="患者ID")
    context: Dict[str, Any] = Field(default_factory=dict, description="上下文信息")
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="对话历史"
    )
    max_iterations: int = Field(default=5, description="最大迭代次数")
    enable_tools: List[str] = Field(
        default_factory=lambda: ["all"],
        description="启用的工具列表"
    )


class AgentAction(BaseModel):
    """Agent动作记录"""
    step: int = Field(..., description="步骤编号")
    thought: str = Field(..., description="思考过程")
    action: str = Field(..., description="动作名称")
    action_input: Dict[str, Any] = Field(..., description="动作输入")
    observation: str = Field(..., description="观察结果")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")


class AgentOutput(BaseModel):
    """Agent输出模型"""
    success: bool = Field(..., description="是否成功")
    answer: str = Field(..., description="最终答案")
    mode: AgentMode = Field(..., description="运行模式")
    actions: List[AgentAction] = Field(default_factory=list, description="动作历史")
    tools_used: List[str] = Field(default_factory=list, description="使用的工具")
    confidence: float = Field(default=0.0, description="置信度", ge=0.0, le=1.0)
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="信息来源")
    recommendations: List[str] = Field(default_factory=list, description="建议")
    warnings: List[str] = Field(default_factory=list, description="警告")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    error_message: Optional[str] = Field(None, description="错误信息")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")


class MedicalAgent:
    """
    医疗顾问Agent
    整合知识图谱查询、向量搜索、患者信息等工具
    提供智能化的医疗咨询和决策支持
    """
    
    def __init__(
        self,
        kg_retriever: Optional[Any] = None,
        vector_retriever: Optional[Any] = None,
        patient_database: Optional[Any] = None,
        llm: Optional[Any] = None,
        use_mock: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化医疗顾问Agent
        
        Args:
            kg_retriever: 知识图谱检索器
            vector_retriever: 向量检索器
            patient_database: 患者数据库
            llm: 大语言模型
            use_mock: 是否使用模拟数据
            config: 配置字典
        """
        self.llm = llm
        self.use_mock = use_mock
        self.config = config or {}
        
        # 初始化工具
        self.tools = self._initialize_tools(
            kg_retriever,
            vector_retriever,
            patient_database
        )
        
        # Agent状态
        self.current_step = 0
        self.actions_history: List[AgentAction] = []
        
        logger.info(f"医疗顾问Agent初始化完成，可用工具: {list(self.tools.keys())}")
    
    def _initialize_tools(
        self,
        kg_retriever: Optional[Any],
        vector_retriever: Optional[Any],
        patient_database: Optional[Any]
    ) -> Dict[str, Any]:
        """初始化工具集"""
        tools = {}
        
        # 知识图谱查询工具
        tools["kg_query"] = KGQueryTool(
            kg_retriever=kg_retriever,
            use_mock=self.use_mock
        )
        
        # 向量搜索工具
        tools["vector_search"] = VectorSearchTool(
            vector_retriever=vector_retriever,
            use_mock=self.use_mock
        )
        
        # 患者信息工具
        tools["patient_info"] = PatientInfoTool(
            patient_database=patient_database,
            use_mock=self.use_mock
        )
        
        return tools
    
    def run(self, input_data: AgentInput) -> AgentOutput:
        """
        运行Agent
        
        Args:
            input_data: Agent输入
            
        Returns:
            Agent输出
        """
        logger.info(f"Agent开始运行: mode={input_data.mode}, query='{input_data.query[:50]}...'")
        
        # 重置状态
        self.current_step = 0
        self.actions_history = []
        
        try:
            # 根据模式选择处理策略
            if input_data.mode == AgentMode.DIAGNOSIS:
                output = self._run_diagnosis_mode(input_data)
            elif input_data.mode == AgentMode.TREATMENT:
                output = self._run_treatment_mode(input_data)
            elif input_data.mode == AgentMode.QA:
                output = self._run_qa_mode(input_data)
            else:  # CONSULTATION
                output = self._run_consultation_mode(input_data)
            
            logger.info(f"Agent运行完成: 使用了{len(output.tools_used)}个工具, {len(output.actions)}个步骤")
            return output
        
        except Exception as e:
            logger.error(f"Agent运行失败: {e}")
            return AgentOutput(
                success=False,
                answer="抱歉，处理您的请求时遇到了问题。",
                mode=input_data.mode,
                actions=self.actions_history,
                tools_used=[],
                error_message=str(e)
            )
    
    def _run_consultation_mode(self, input_data: AgentInput) -> AgentOutput:
        """咨询模式"""
        logger.info("运行咨询模式")
        
        # 步骤1: 分析查询意图
        intent = self._analyze_query_intent(input_data.query)
        
        # 步骤2: 获取患者信息（如果有患者ID）
        patient_info = None
        if input_data.patient_id:
            patient_info = self._get_patient_context(input_data.patient_id)
        
        # 步骤3: 检索相关知识
        knowledge = self._retrieve_knowledge(input_data.query, intent)
        
        # 步骤4: 生成答案
        answer = self._generate_answer(
            input_data.query,
            intent,
            knowledge,
            patient_info
        )
        
        # 步骤5: 生成建议和警告
        recommendations = self._generate_recommendations(intent, patient_info)
        warnings = self._generate_warnings(patient_info)
        
        # 计算置信度
        confidence = self._calculate_confidence(knowledge)
        
        return AgentOutput(
            success=True,
            answer=answer,
            mode=input_data.mode,
            actions=self.actions_history,
            tools_used=self._get_tools_used(),
            confidence=confidence,
            sources=knowledge,
            recommendations=recommendations,
            warnings=warnings,
            metadata={"intent": intent}
        )
    
    def _run_diagnosis_mode(self, input_data: AgentInput) -> AgentOutput:
        """诊断模式"""
        logger.info("运行诊断模式")
        
        # 步骤1: 获取患者信息
        patient_info = None
        if input_data.patient_id:
            patient_info = self._get_patient_context(input_data.patient_id)
            self._record_action(
                thought="需要获取患者的详细信息以辅助诊断",
                action="patient_info",
                action_input={"patient_id": input_data.patient_id},
                observation=f"已获取患者信息，发现{len(patient_info.get('allergies', []))}项过敏史"
            )
        
        # 步骤2: 分析症状
        symptoms = self._extract_symptoms(input_data.query, input_data.context)
        
        # 步骤3: 查询可能的疾病
        possible_diseases = self._query_diseases_by_symptoms(symptoms)
        self._record_action(
            thought=f"根据症状'{', '.join(symptoms)}'查询可能的疾病",
            action="kg_query",
            action_input={"symptoms": symptoms},
            observation=f"找到{len(possible_diseases)}个可能的疾病"
        )
        
        # 步骤4: 搜索诊断相关信息
        diagnosis_info = self._search_diagnosis_info(symptoms, possible_diseases)
        
        # 步骤5: 生成诊断建议
        answer = self._generate_diagnosis_answer(
            symptoms,
            possible_diseases,
            diagnosis_info,
            patient_info
        )
        
        recommendations = [
            "建议前往医院进行专业诊断",
            "如症状加重，请立即就医",
            "本诊断仅供参考，不能替代医生诊断"
        ]
        
        return AgentOutput(
            success=True,
            answer=answer,
            mode=input_data.mode,
            actions=self.actions_history,
            tools_used=self._get_tools_used(),
            confidence=0.7,
            sources=diagnosis_info,
            recommendations=recommendations,
            warnings=self._generate_warnings(patient_info)
        )
    
    def _run_treatment_mode(self, input_data: AgentInput) -> AgentOutput:
        """治疗模式"""
        logger.info("运行治疗模式")
        
        # 步骤1: 提取疾病信息
        disease = self._extract_disease(input_data.query, input_data.context)
        
        # 步骤2: 获取患者信息
        patient_info = None
        if input_data.patient_id:
            patient_info = self._get_patient_context(input_data.patient_id)
        
        # 步骤3: 查询治疗方案
        treatment_info = self._query_treatment_options(disease)
        self._record_action(
            thought=f"查询{disease}的治疗方案",
            action="vector_search",
            action_input={"query": f"{disease}的治疗方法"},
            observation=f"找到{len(treatment_info)}条治疗信息"
        )
        
        # 步骤4: 查询推荐药品
        drug_info = self._query_recommended_drugs(disease, patient_info)
        
        # 步骤5: 生成治疗建议
        answer = self._generate_treatment_answer(
            disease,
            treatment_info,
            drug_info,
            patient_info
        )
        
        recommendations = [
            "请在医生指导下进行治疗",
            "定期复诊，监测治疗效果",
            "如有不适，及时与医生沟通"
        ]
        
        warnings = []
        if patient_info and patient_info.get("allergies"):
            warnings.append(f"患者对{', '.join(patient_info['allergies'])}过敏，用药需谨慎")
        
        return AgentOutput(
            success=True,
            answer=answer,
            mode=input_data.mode,
            actions=self.actions_history,
            tools_used=self._get_tools_used(),
            confidence=0.8,
            sources=treatment_info + drug_info,
            recommendations=recommendations,
            warnings=warnings
        )
    
    def _run_qa_mode(self, input_data: AgentInput) -> AgentOutput:
        """问答模式"""
        logger.info("运行问答模式")
        
        # 步骤1: 向量搜索相关信息
        search_results = self._search_relevant_info(input_data.query)
        self._record_action(
            thought="搜索与问题相关的医疗知识",
            action="vector_search",
            action_input={"query": input_data.query},
            observation=f"找到{len(search_results)}条相关信息"
        )
        
        # 步骤2: 知识图谱补充
        kg_results = self._query_kg_supplement(input_data.query)
        
        # 步骤3: 生成答案
        answer = self._generate_qa_answer(
            input_data.query,
            search_results,
            kg_results
        )
        
        return AgentOutput(
            success=True,
            answer=answer,
            mode=input_data.mode,
            actions=self.actions_history,
            tools_used=self._get_tools_used(),
            confidence=0.75,
            sources=search_results + kg_results,
            recommendations=["如需详细信息，建议咨询专业医生"],
            warnings=[]
        )
    
    def _analyze_query_intent(self, query: str) -> str:
        """分析查询意图"""
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ["症状", "表现", "感觉", "不舒服"]):
            return "symptom_inquiry"
        elif any(kw in query_lower for kw in ["是什么", "什么病", "疾病"]):
            return "disease_info"
        elif any(kw in query_lower for kw in ["治疗", "怎么办", "如何", "方法"]):
            return "treatment"
        elif any(kw in query_lower for kw in ["药", "吃什么", "用药"]):
            return "medication"
        elif any(kw in query_lower for kw in ["预防", "避免", "注意"]):
            return "prevention"
        else:
            return "general"
    
    def _get_patient_context(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """获取患者上下文信息"""
        try:
            from .tools import PatientInfoInput
            
            input_data = PatientInfoInput(
                patient_id=patient_id,
                info_type="all",
                include_history=True,
                max_history_records=3
            )
            
            output = self.tools["patient_info"].run(input_data)
            
            if output.success:
                return {
                    "basic_info": output.basic_info,
                    "profile": output.profile,
                    "allergies": output.allergies,
                    "current_medications": output.current_medications,
                    "medical_history": output.medical_history
                }
            return None
        except Exception as e:
            logger.error(f"获取患者信息失败: {e}")
            return None
    
    def _retrieve_knowledge(
        self,
        query: str,
        intent: str
    ) -> List[Dict[str, Any]]:
        """检索相关知识"""
        knowledge = []
        
        try:
            # 向量搜索
            from .tools import VectorSearchInput
            
            search_input = VectorSearchInput(
                query_text=query,
                top_k=3,
                score_threshold=0.6
            )
            
            search_output = self.tools["vector_search"].run(search_input)
            
            if search_output.success:
                for result in search_output.results:
                    knowledge.append({
                        "source": "vector_search",
                        "content": result.get("text", ""),
                        "score": result.get("score", 0.0)
                    })
            
            self._record_action(
                thought=f"根据意图'{intent}'检索相关知识",
                action="vector_search",
                action_input={"query": query},
                observation=f"找到{len(knowledge)}条相关知识"
            )
        
        except Exception as e:
            logger.error(f"知识检索失败: {e}")
        
        return knowledge
    
    def _generate_answer(
        self,
        query: str,
        intent: str,
        knowledge: List[Dict[str, Any]],
        patient_info: Optional[Dict[str, Any]]
    ) -> str:
        """生成答案"""
        if self.llm and not self.use_mock:
            # 使用LLM生成答案
            # TODO: 实现LLM调用
            logger.warning("LLM未配置，使用模拟答案生成")
        
        # 模拟答案生成
        if not knowledge:
            return "抱歉，我没有找到相关信息。建议您咨询专业医生获取帮助。"
        
        # 基于知识构建答案
        answer_parts = ["根据医疗知识库的信息：\n"]
        
        for i, item in enumerate(knowledge[:3], 1):
            answer_parts.append(f"{i}. {item.get('content', '')}\n")
        
        # 添加患者特定建议
        if patient_info and patient_info.get("allergies"):
            answer_parts.append(f"\n注意：您对{', '.join(patient_info['allergies'])}过敏，相关治疗需避免使用。")
        
        answer_parts.append("\n以上信息仅供参考，具体诊疗请咨询医生。")
        
        return "".join(answer_parts)
    
    def _generate_recommendations(
        self,
        intent: str,
        patient_info: Optional[Dict[str, Any]]
    ) -> List[str]:
        """生成建议"""
        recommendations = [
            "如有疑问，请及时咨询专业医生",
            "定期进行健康检查"
        ]
        
        if intent == "symptom_inquiry":
            recommendations.insert(0, "如症状持续或加重，建议尽快就医")
        elif intent == "medication":
            recommendations.insert(0, "用药前请仔细阅读说明书或咨询药师")
        
        return recommendations
    
    def _generate_warnings(
        self,
        patient_info: Optional[Dict[str, Any]]
    ) -> List[str]:
        """生成警告"""
        warnings = []
        
        if patient_info:
            if patient_info.get("allergies"):
                warnings.append(f"过敏警告: {', '.join(patient_info['allergies'])}")
            
            if patient_info.get("profile", {}).get("chronic_diseases"):
                chronic = patient_info["profile"]["chronic_diseases"]
                warnings.append(f"慢性疾病: {', '.join(chronic)}")
        
        return warnings
    
    def _calculate_confidence(self, knowledge: List[Dict[str, Any]]) -> float:
        """计算置信度"""
        if not knowledge:
            return 0.3
        
        avg_score = sum(k.get("score", 0.0) for k in knowledge) / len(knowledge)
        knowledge_factor = min(len(knowledge) / 5.0, 1.0)
        
        confidence = avg_score * 0.7 + knowledge_factor * 0.3
        return round(confidence, 2)
    
    def _record_action(
        self,
        thought: str,
        action: str,
        action_input: Dict[str, Any],
        observation: str
    ) -> None:
        """记录Agent动作"""
        self.current_step += 1
        
        action_record = AgentAction(
            step=self.current_step,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation
        )
        
        self.actions_history.append(action_record)
    
    def _get_tools_used(self) -> List[str]:
        """获取使用过的工具列表"""
        tools_used = set()
        for action in self.actions_history:
            tools_used.add(action.action)
        return list(tools_used)
    def _extract_symptoms(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """从查询中提取症状"""
        # 简单的症状提取（实际应使用NER）
        common_symptoms = [
            "头痛", "发热", "咳嗽", "流涕", "咽痛", "胸痛", "腹痛",
            "恶心", "呕吐", "腹泻", "头晕", "乏力", "失眠", "心悸"
        ]
        
        symptoms = []
        query_lower = query.lower()
        
        for symptom in common_symptoms:
            if symptom in query_lower:
                symptoms.append(symptom)
        
        # 从上下文中提取
        if "symptoms" in context:
            symptoms.extend(context["symptoms"])
        
        return list(set(symptoms))  # 去重
    
    def _query_diseases_by_symptoms(self, symptoms: List[str]) -> List[Dict[str, Any]]:
        """根据症状查询可能的疾病"""
        from .tools import KGQueryInput
        
        try:
            query_input = KGQueryInput(
                query_type="entity",
                entity_type="Disease",
                max_results=5
            )
            
            output = self.tools["kg_query"].run(query_input)
            
            if output.success:
                return output.results
            return []
        except Exception as e:
            logger.error(f"查询疾病失败: {e}")
            return []
    
    def _search_diagnosis_info(
        self,
        symptoms: List[str],
        diseases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """搜索诊断相关信息"""
        from .tools import VectorSearchInput
        
        query_text = f"症状: {', '.join(symptoms)}, 可能的疾病及诊断方法"
        
        try:
            search_input = VectorSearchInput(
                query_text=query_text,
                top_k=3
            )
            
            output = self.tools["vector_search"].run(search_input)
            
            if output.success:
                return output.results
            return []
        except Exception as e:
            logger.error(f"搜索诊断信息失败: {e}")
            return []
    
    def _generate_diagnosis_answer(
        self,
        symptoms: List[str],
        diseases: List[Dict[str, Any]],
        diagnosis_info: List[Dict[str, Any]],
        patient_info: Optional[Dict[str, Any]]
    ) -> str:
        """生成诊断答案"""
        answer_parts = [f"根据您提供的症状：{', '.join(symptoms)}\n\n"]
        
        if diseases:
            answer_parts.append("可能的疾病包括：\n")
            for i, disease in enumerate(diseases[:3], 1):
                disease_name = disease.get("properties", {}).get("name", "未知疾病")
                answer_parts.append(f"{i}. {disease_name}\n")
        
        if diagnosis_info:
            answer_parts.append("\n相关诊断信息：\n")
            for info in diagnosis_info[:2]:
                answer_parts.append(f"• {info.get('text', '')[:100]}...\n")
        
        answer_parts.append("\n⚠️ 重要提示：\n")
        answer_parts.append("- 以上仅为初步分析，不能替代专业医生诊断\n")
        answer_parts.append("- 建议前往医院进行详细检查\n")
        answer_parts.append("- 如症状严重或持续加重，请立即就医\n")
        
        return "".join(answer_parts)
    
    def _extract_disease(self, query: str, context: Dict[str, Any]) -> str:
        """提取疾病名称"""
        # 从上下文中提取
        if "disease" in context:
            return context["disease"]
        
        # 从查询中提取（简单匹配）
        common_diseases = ["高血压", "糖尿病", "感冒", "发烧", "咳嗽", "头痛"]
        
        for disease in common_diseases:
            if disease in query:
                return disease
        
        return "未指定疾病"
    
    def _query_treatment_options(self, disease: str) -> List[Dict[str, Any]]:
        """查询治疗方案"""
        from .tools import VectorSearchInput
        
        try:
            search_input = VectorSearchInput(
                query_text=f"{disease}的治疗方法和用药建议",
                top_k=5,
                filter_category="treatment"
            )
            
            output = self.tools["vector_search"].run(search_input)
            
            if output.success:
                return output.results
            return []
        except Exception as e:
            logger.error(f"查询治疗方案失败: {e}")
            return []
    
    def _query_recommended_drugs(
        self,
        disease: str,
        patient_info: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """查询推荐药品"""
        from .tools import KGQueryInput
        
        try:
            query_input = KGQueryInput(
                query_type="relationship",
                relationship_type="TREATED_BY",
                max_results=5
            )
            
            output = self.tools["kg_query"].run(query_input)
            
            if output.success:
                return output.results
            return []
        except Exception as e:
            logger.error(f"查询推荐药品失败: {e}")
            return []
    
    def _generate_treatment_answer(
        self,
        disease: str,
        treatment_info: List[Dict[str, Any]],
        drug_info: List[Dict[str, Any]],
        patient_info: Optional[Dict[str, Any]]
    ) -> str:
        """生成治疗答案"""
        answer_parts = [f"针对{disease}的治疗建议：\n\n"]
        
        if treatment_info:
            answer_parts.append("【治疗方案】\n")
            for i, info in enumerate(treatment_info[:3], 1):
                text = info.get("text", "")
                answer_parts.append(f"{i}. {text[:150]}...\n")
        
        if drug_info:
            answer_parts.append("\n【推荐药品】\n")
            for i, drug in enumerate(drug_info[:3], 1):
                drug_name = drug.get("properties", {}).get("name", "未知药品")
                answer_parts.append(f"{i}. {drug_name}\n")
        
        # 患者特定提醒
        if patient_info:
            answer_parts.append("\n【个人化提醒】\n")
            if patient_info.get("allergies"):
                answer_parts.append(f"⚠️ 您对{', '.join(patient_info['allergies'])}过敏，用药时需避免\n")
            if patient_info.get("current_medications"):
                answer_parts.append(f"ℹ️ 您当前正在使用其他药物，请注意药物相互作用\n")
        
        answer_parts.append("\n📌 重要说明：\n")
        answer_parts.append("- 请在医生指导下进行治疗\n")
        answer_parts.append("- 不要自行调整药物剂量\n")
        answer_parts.append("- 定期复诊，监测治疗效果\n")
        
        return "".join(answer_parts)
    
    def _search_relevant_info(self, query: str) -> List[Dict[str, Any]]:
        """搜索相关信息"""
        from .tools import VectorSearchInput
        
        try:
            search_input = VectorSearchInput(
                query_text=query,
                top_k=5,
                include_metadata=True
            )
            
            output = self.tools["vector_search"].run(search_input)
            
            if output.success:
                return output.results
            return []
        except Exception as e:
            logger.error(f"搜索相关信息失败: {e}")
            return []
    
    def _query_kg_supplement(self, query: str) -> List[Dict[str, Any]]:
        """知识图谱补充查询"""
        from .tools import KGQueryInput
        
        try:
            query_input = KGQueryInput(
                query_type="entity",
                max_results=3
            )
            
            output = self.tools["kg_query"].run(query_input)
            
            if output.success:
                return output.results
            return []
        except Exception as e:
            logger.error(f"知识图谱补充查询失败: {e}")
            return []
    
    def _generate_qa_answer(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        kg_results: List[Dict[str, Any]]
    ) -> str:
        """生成问答答案"""
        if not search_results and not kg_results:
            return "抱歉，我没有找到相关信息。建议您：\n1. 换个方式提问\n2. 咨询专业医生\n3. 查阅权威医疗网站"
        
        answer_parts = [f"关于您的问题「{query}」：\n\n"]
        
        if search_results:
            answer_parts.append("【相关信息】\n")
            for i, result in enumerate(search_results[:3], 1):
                text = result.get("text", "")
                score = result.get("score", 0.0)
                answer_parts.append(f"{i}. {text}\n")
                answer_parts.append(f"   (相关度: {score:.0%})\n\n")
        
        if kg_results:
            answer_parts.append("【知识图谱】\n")
            for kg_result in kg_results[:2]:
                entity_name = kg_result.get("properties", {}).get("name", "")
                if entity_name:
                    answer_parts.append(f"• {entity_name}\n")
        
        answer_parts.append("\n💡 温馨提示：\n")
        answer_parts.append("本回答仅供参考，具体情况请咨询专业医生。\n")
        
        return "".join(answer_parts)
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """
        获取指定工具
        
        Args:
            tool_name: 工具名称
            
        Returns:
            工具实例
        """
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[Dict[str, str]]:
        """
        列出所有可用工具
        
        Returns:
            工具列表
        """
        tools_info = []
        for name, tool in self.tools.items():
            tools_info.append({
                "name": name,
                "description": tool.description
            })
        return tools_info
    
    def reset(self) -> None:
        """重置Agent状态"""
        self.current_step = 0
        self.actions_history = []
        logger.info("Agent状态已重置")


def create_medical_agent(
    use_mock: bool = True,
    config: Optional[Dict[str, Any]] = None
) -> MedicalAgent:
    """
    便捷函数：创建医疗顾问Agent
    
    Args:
        use_mock: 是否使用模拟数据
        config: 配置字典
        
    Returns:
        MedicalAgent实例
    """
    return MedicalAgent(
        use_mock=use_mock,
        config=config
    )