"""
问答相关提示词模板
"""

from typing import List, Optional, Dict, Any


class QAPrompts:
    """问答提示词模板集合"""
    
    @staticmethod
    def get_qa_prompt(
        question: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        生成问答提示词
        
        Args:
            question: 用户问题
            context: 上下文信息
            conversation_history: 对话历史
            
        Returns:
            提示词字符串
        """
        prompt = """你是一位专业的医疗顾问，负责回答患者的健康相关问题。

"""
        
        if context:
            prompt += f"""【相关知识】
{context}

"""
        
        if conversation_history:
            prompt += "【对话历史】\n"
            for conv in conversation_history[-3:]:  # 只保留最近3轮
                prompt += f"Q: {conv.get('question', '')}\n"
                prompt += f"A: {conv.get('answer', '')}\n\n"
        
        prompt += f"""【当前问题】
{question}

【回答要求】
1. 基于提供的医疗知识回答
2. 使用通俗易懂的语言
3. 结构清晰、重点突出
4. 如涉及诊断或治疗，强调需咨询医生
5. 提供实用的建议
6. 避免过度承诺或保证

请专业、准确地回答问题。
"""
        return prompt
    
    @staticmethod
    def get_medical_explanation_prompt(
        term: str,
        level: str = "patient"
    ) -> str:
        """
        生成医学术语解释提示词
        
        Args:
            term: 医学术语
            level: 解释级别 (patient/professional)
            
        Returns:
            提示词字符串
        """
        if level == "patient":
            prompt = f"""请用通俗的语言解释以下医学术语。

【术语】
{term}

【解释要求】
1. 避免使用复杂的医学术语
2. 用日常语言和比喻说明
3. 举例说明实际含义
4. 说明为什么重要
5. 患者需要知道什么

请确保普通人能够理解。
"""
        else:
            prompt = f"""请提供以下医学术语的专业解释。

【术语】
{term}

【解释内容】
1. 定义和概念
2. 病理生理基础
3. 临床意义
4. 相关术语
5. 最新研究进展

请提供专业的医学解释。
"""
        return prompt
    
    @staticmethod
    def get_symptom_inquiry_prompt(
        symptom: str,
        additional_info: Optional[str] = None
    ) -> str:
        """
        生成症状咨询提示词
        
        Args:
            symptom: 症状描述
            additional_info: 额外信息
            
        Returns:
            提示词字符串
        """
        prompt = f"""患者咨询症状问题，请提供专业建议。

【症状】
{symptom}

"""
        
        if additional_info:
            prompt += f"【额外信息】\n{additional_info}\n\n"
        
        prompt += """【回答内容】
1. 该症状可能的原因（从常见到罕见）
2. 需要关注的伴随症状
3. 何时需要就医（紧急情况的判断）
4. 自我缓解的方法
5. 就医时应该提供的信息
6. 预防措施

【注意】
- 不要做出明确诊断
- 强调就医的重要性
- 提供实用的建议

请提供有帮助的回答。
"""
        return prompt
    
    @staticmethod
    def get_prevention_advice_prompt(
        disease: str,
        risk_factors: Optional[List[str]] = None
    ) -> str:
        """
        生成预防建议提示词
        
        Args:
            disease: 疾病名称
            risk_factors: 风险因素
            
        Returns:
            提示词字符串
        """
        prompt = f"""请提供关于预防{disease}的建议。

【疾病】
{disease}

"""
        
        if risk_factors:
            prompt += f"【风险因素】\n{', '.join(risk_factors)}\n\n"
        
        prompt += """【预防建议】
1. 一级预防（预防疾病发生）：
   - 生活方式调整
   - 饮食建议
   - 运动指导
   - 环境因素控制

2. 二级预防（早期发现）：
   - 筛查建议
   - 自我监测
   - 定期检查

3. 高危人群特别注意事项

4. 疫苗接种（如适用）

5. 具体可行的行动计划

请提供实用的预防指导。
"""
        return prompt
    
    @staticmethod
    def get_health_education_prompt(
        topic: str,
        target_audience: str = "general"
    ) -> str:
        """
        生成健康教育提示词
        
        Args:
            topic: 教育主题
            target_audience: 目标受众
            
        Returns:
            提示词字符串
        """
        prompt = f"""请提供关于"{topic}"的健康教育内容。

【目标受众】
{target_audience}

【内容要求】
1. 核心知识点（3-5个）
2. 常见误区澄清
3. 实用建议
4. 自我管理技巧
5. 何时寻求专业帮助
6. 推荐的可靠信息来源

【呈现方式】
- 清晰的结构
- 要点突出
- 易于理解和记忆
- 包含实例说明

请提供有教育意义的内容。
"""
        return prompt
    
    @staticmethod
    def get_medication_guidance_prompt(
        medication: str,
        patient_concerns: Optional[List[str]] = None
    ) -> str:
        """
        生成用药指导提示词
        
        Args:
            medication: 药物名称
            patient_concerns: 患者关注点
            
        Returns:
            提示词字符串
        """
        prompt = f"""患者询问关于药物"{medication}"的使用问题。

"""
        
        if patient_concerns:
            prompt += f"【患者关注】\n{chr(10).join(f'- {c}' for c in patient_concerns)}\n\n"
        
        prompt += """【指导内容】
1. 药物基本信息：
   - 作用和用途
   - 正确用法
   - 最佳服用时间

2. 常见问题：
   - 漏服怎么办
   - 能否与其他药物同服
   - 能否与食物同服

3. 副作用管理：
   - 常见副作用
   - 如何应对
   - 何时需要就医

4. 注意事项：
   - 禁忌情况
   - 特殊人群
   - 储存方法

5. 疗程和停药：
   - 需要服用多久
   - 能否自行停药
   - 停药注意事项

请提供清晰的用药指导。
"""
        return prompt
    
    @staticmethod
    def get_emergency_guidance_prompt(
        situation: str
    ) -> str:
        """
        生成紧急情况指导提示词
        
        Args:
            situation: 紧急情况描述
            
        Returns:
            提示词字符串
        """
        prompt = f"""【紧急情况咨询】

{situation}

【紧急响应】
请立即提供：

1. 紧急程度评估（1-5级）
2. 立即应采取的措施（步骤清晰）
3. 何时拨打急救电话
4. 等待救援时的注意事项
5. 不应该做的事情

⚠️ 这是紧急情况，请：
- 快速给出明确指导
- 使用简单直接的语言
- 强调安全第一
- 建议立即就医（如需要）

请提供紧急指导。
"""
        return prompt
    
    @staticmethod
    def get_follow_up_question_prompt(
        original_question: str,
        original_answer: str,
        follow_up: str
    ) -> str:
        """
        生成追问回答提示词
        
        Args:
            original_question: 原始问题
            original_answer: 原始回答
            follow_up: 追问内容
            
        Returns:
            提示词字符串
        """
        prompt = f"""这是一个追问，请基于之前的对话回答。

【原始问题】
{original_question}

【之前的回答】
{original_answer}

【追问】
{follow_up}

【回答要求】
1. 保持与之前回答的一致性
2. 针对追问的具体内容
3. 提供更详细的信息
4. 必要时修正或补充之前的回答

请回答追问。
"""
        return prompt