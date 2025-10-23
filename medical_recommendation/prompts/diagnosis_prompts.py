"""
诊断相关提示词模板
"""

from typing import Dict, List, Any, Optional


class DiagnosisPrompts:
    """诊断提示词模板集合"""
    
    @staticmethod
    def get_diagnosis_prompt(
        symptoms: List[str],
        duration: Optional[str] = None,
        severity: Optional[str] = None,
        patient_history: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """
        生成诊断提示词
        
        Args:
            symptoms: 症状列表
            duration: 持续时间
            severity: 严重程度
            patient_history: 病史
            context: 额外上下文
            
        Returns:
            提示词字符串
        """
        prompt = f"""你是一位经验丰富的医生，需要基于患者的症状进行初步诊断分析。

【患者症状】
{', '.join(symptoms)}

"""
        
        if duration:
            prompt += f"【持续时间】\n{duration}\n\n"
        
        if severity:
            prompt += f"【严重程度】\n{severity}\n\n"
        
        if patient_history:
            prompt += f"【病史信息】\n{patient_history}\n\n"
        
        if context:
            prompt += f"【相关信息】\n{context}\n\n"
        
        prompt += """【诊断要求】
1. 根据症状列出最可能的3-5个疾病诊断
2. 为每个诊断提供置信度评分（0-1）
3. 说明诊断依据和推理过程
4. 列出需要进一步检查的项目
5. 评估紧急程度（normal/urgent/emergency）

【输出格式】
请按以下格式输出：

主要诊断：[疾病名称]
置信度：[0-1之间的数值]
诊断依据：[详细说明]

鉴别诊断：
1. [疾病名称] - 置信度：[数值] - 理由：[说明]
2. [疾病名称] - 置信度：[数值] - 理由：[说明]
...

建议检查：
- [检查项目1]
- [检查项目2]
...

紧急程度：[normal/urgent/emergency]

注意事项：
- [重要提醒]
"""
        
        return prompt
    
    @staticmethod
    def get_differential_diagnosis_prompt(
        primary_diagnosis: str,
        symptoms: List[str],
        similar_diseases: List[str]
    ) -> str:
        """
        生成鉴别诊断提示词
        
        Args:
            primary_diagnosis: 主要诊断
            symptoms: 症状列表
            similar_diseases: 相似疾病列表
            
        Returns:
            提示词字符串
        """
        prompt = f"""作为医学专家，请进行鉴别诊断分析。

【初步诊断】
{primary_diagnosis}

【患者症状】
{', '.join(symptoms)}

【需要鉴别的疾病】
{', '.join(similar_diseases)}

【分析要求】
1. 比较各疾病的典型症状特征
2. 指出当前症状更支持哪个诊断
3. 说明需要哪些检查来明确诊断
4. 评估每个疾病的可能性

请提供详细的鉴别诊断分析。
"""
        return prompt
    
    @staticmethod
    def get_symptom_analysis_prompt(
        symptom: str,
        related_symptoms: Optional[List[str]] = None
    ) -> str:
        """
        生成症状分析提示词
        
        Args:
            symptom: 主要症状
            related_symptoms: 相关症状
            
        Returns:
            提示词字符串
        """
        prompt = f"""请分析以下症状的可能原因和相关疾病。

【主要症状】
{symptom}

"""
        
        if related_symptoms:
            prompt += f"【伴随症状】\n{', '.join(related_symptoms)}\n\n"
        
        prompt += """【分析内容】
1. 该症状的常见原因（从最常见到罕见）
2. 可能关联的疾病
3. 需要警惕的危险信号
4. 建议的处理方式

请提供专业的医学分析。
"""
        return prompt
    
    @staticmethod
    def get_risk_assessment_prompt(
        patient_profile: Dict[str, Any],
        diagnosis: str
    ) -> str:
        """
        生成风险评估提示词
        
        Args:
            patient_profile: 患者档案
            diagnosis: 诊断
            
        Returns:
            提示词字符串
        """
        age = patient_profile.get('age', 'unknown')
        gender = patient_profile.get('gender', 'unknown')
        chronic_diseases = patient_profile.get('chronic_conditions', [])
        risk_factors = patient_profile.get('risk_factors', [])
        
        prompt = f"""请评估该患者的健康风险。

【患者信息】
年龄：{age}
性别：{gender}
慢性病：{', '.join(chronic_diseases) if chronic_diseases else '无'}
风险因素：{', '.join(risk_factors) if risk_factors else '无'}

【当前诊断】
{diagnosis}

【评估要求】
1. 评估疾病进展风险
2. 识别并发症风险
3. 预后评估
4. 生活质量影响
5. 提出风险管理建议

请提供全面的风险评估报告。
"""
        return prompt
    
    @staticmethod
    def get_emergency_assessment_prompt(
        symptoms: List[str],
        vital_signs: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        生成紧急情况评估提示词
        
        Args:
            symptoms: 症状列表
            vital_signs: 生命体征
            
        Returns:
            提示词字符串
        """
        prompt = f"""【紧急情况评估】

患者症状：
{', '.join(symptoms)}

"""
        
        if vital_signs:
            prompt += f"""生命体征：
- 体温：{vital_signs.get('temperature', 'N/A')}
- 血压：{vital_signs.get('blood_pressure', 'N/A')}
- 心率：{vital_signs.get('heart_rate', 'N/A')}
- 呼吸频率：{vital_signs.get('respiratory_rate', 'N/A')}

"""
        
        prompt += """请立即评估：
1. 是否需要紧急医疗干预？
2. 紧急程度分级（1-5级，5级最紧急）
3. 立即应采取的措施
4. 如何安全转运患者

这是紧急评估，请快速给出明确建议。
"""
        return prompt


    @staticmethod
    def get_diagnosis_explanation_prompt(
        diagnosis: str,
        for_patient: bool = True
    ) -> str:
        """
        生成诊断解释提示词
        
        Args:
            diagnosis: 诊断名称
            for_patient: 是否面向患者（否则面向医生）
            
        Returns:
            提示词字符串
        """
        if for_patient:
            prompt = f"""请用通俗易懂的语言向患者解释以下诊断。

【诊断】
{diagnosis}

【解释要求】
1. 用简单的语言说明这是什么疾病
2. 解释可能的病因
3. 说明常见的症状表现
4. 解释疾病的自然进程
5. 告知预后和可能的结果
6. 避免使用过多医学术语

请确保患者能够理解。
"""
        else:
            prompt = f"""请为医疗专业人员提供关于以下诊断的详细说明。

【诊断】
{diagnosis}

【说明内容】
1. 疾病的病理生理机制
2. 流行病学特征
3. 诊断标准和依据
4. 鉴别诊断要点
5. 治疗原则
6. 预后评估因素

请提供专业的医学解释。
"""
        return prompt