"""
推荐相关提示词模板
"""

from typing import Dict, List, Any, Optional


class RecommendationPrompts:
    """推荐提示词模板集合"""
    
    @staticmethod
    def get_drug_recommendation_prompt(
        diagnosis: str,
        symptoms: List[str],
        patient_profile: Optional[Dict[str, Any]] = None,
        contraindications: Optional[List[str]] = None,
        current_medications: Optional[List[str]] = None
    ) -> str:
        """
        生成药品推荐提示词
        
        Args:
            diagnosis: 诊断
            symptoms: 症状
            patient_profile: 患者档案
            contraindications: 禁忌症
            current_medications: 当前用药
            
        Returns:
            提示词字符串
        """
        prompt = f"""作为临床药师，请为以下情况推荐合适的药品。

【诊断】
{diagnosis}

【症状】
{', '.join(symptoms)}

"""
        
        if patient_profile:
            age = patient_profile.get('age', 'unknown')
            gender = patient_profile.get('gender', 'unknown')
            allergies = patient_profile.get('allergies', [])
            
            prompt += f"""【患者信息】
年龄：{age}
性别：{gender}
过敏史：{', '.join(allergies) if allergies else '无'}

"""
        
        if contraindications:
            prompt += f"【禁忌症】\n{', '.join(contraindications)}\n\n"
        
        if current_medications:
            prompt += f"【当前用药】\n{', '.join(current_medications)}\n\n"
        
        prompt += """【推荐要求】
1. 推荐3-5种合适的药品（优先级排序）
2. 说明每种药品的：
   - 药品名称（通用名和商品名）
   - 推荐剂量和用法
   - 疗程建议
   - 主要作用机制
   - 常见副作用
   - 注意事项
3. 检查药物相互作用
4. 评估用药安全性
5. 提供替代方案

请确保推荐安全、有效、合理。
"""
        return prompt
    
    @staticmethod
    def get_treatment_plan_prompt(
        diagnosis: str,
        severity: str,
        patient_profile: Optional[Dict[str, Any]] = None,
        treatment_goals: Optional[List[str]] = None
    ) -> str:
        """
        生成治疗方案提示词
        
        Args:
            diagnosis: 诊断
            severity: 严重程度
            patient_profile: 患者档案
            treatment_goals: 治疗目标
            
        Returns:
            提示词字符串
        """
        prompt = f"""请制定详细的治疗方案。

【诊断】
{diagnosis}

【严重程度】
{severity}

"""
        
        if patient_profile:
            prompt += f"""【患者信息】
年龄：{patient_profile.get('age', 'unknown')}
慢性病：{', '.join(patient_profile.get('chronic_conditions', [])) or '无'}
生活方式：{patient_profile.get('lifestyle', {})}

"""
        
        if treatment_goals:
            prompt += f"【治疗目标】\n{chr(10).join(f'- {goal}' for goal in treatment_goals)}\n\n"
        
        prompt += """【方案要求】
1. 分阶段治疗计划：
   - 急性期/初始治疗期
   - 调整期
   - 维持期（如适用）
   
2. 每个阶段包括：
   - 持续时间
   - 治疗目标
   - 具体干预措施
   - 用药方案
   - 监测指标
   - 预期结果

3. 非药物治疗：
   - 生活方式调整
   - 饮食建议
   - 运动指导
   - 心理支持

4. 随访计划：
   - 复诊时间
   - 检查项目
   - 调整标准

5. 应急预案：
   - 病情恶化的迹象
   - 紧急处理措施

请提供全面、个性化的治疗方案。
"""
        return prompt
    
    @staticmethod
    def get_lifestyle_recommendation_prompt(
        diagnosis: str,
        patient_profile: Dict[str, Any]
    ) -> str:
        """
        生成生活方式建议提示词
        
        Args:
            diagnosis: 诊断
            patient_profile: 患者档案
            
        Returns:
            提示词字符串
        """
        lifestyle = patient_profile.get('lifestyle', {})
        
        prompt = f"""请为患者提供生活方式调整建议。

【诊断】
{diagnosis}

【当前生活方式】
- 吸烟：{'是' if lifestyle.get('smoking') else '否'}
- 饮酒：{lifestyle.get('drinking', '未知')}
- 运动：{lifestyle.get('exercise', '未知')}
- 饮食习惯：{lifestyle.get('diet', '未知')}

【建议范围】
1. 饮食调整：
   - 推荐的食物
   - 应避免的食物
   - 营养补充
   - 进餐时间和频率

2. 运动指导：
   - 适合的运动类型
   - 运动强度和时长
   - 注意事项
   - 禁忌活动

3. 作息管理：
   - 睡眠建议
   - 工作安排
   - 压力管理

4. 戒除不良习惯：
   - 戒烟计划
   - 限酒建议
   - 其他不良习惯

5. 自我管理：
   - 症状监测
   - 记录建议
   - 预警信号

请提供具体、可执行的建议。
"""
        return prompt
    
    @staticmethod
    def get_alternative_treatment_prompt(
        diagnosis: str,
        standard_treatment: str,
        patient_constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        生成替代治疗方案提示词
        
        Args:
            diagnosis: 诊断
            standard_treatment: 标准治疗
            patient_constraints: 患者限制条件
            
        Returns:
            提示词字符串
        """
        prompt = f"""请提供替代治疗方案。

【诊断】
{diagnosis}

【标准治疗】
{standard_treatment}

"""
        
        if patient_constraints:
            prompt += f"""【限制条件】
{chr(10).join(f'- {k}: {v}' for k, v in patient_constraints.items())}

"""
        
        prompt += """【替代方案要求】
1. 列出2-3种可行的替代方案
2. 每种方案说明：
   - 治疗原理
   - 适用情况
   - 优点和缺点
   - 预期效果
   - 费用对比
3. 方案选择建议
4. 注意事项和风险

请确保方案的科学性和可行性。
"""
        return prompt
    
    @staticmethod
    def get_medication_adjustment_prompt(
        current_medication: str,
        current_dosage: str,
        treatment_response: str,
        side_effects: Optional[List[str]] = None
    ) -> str:
        """
        生成用药调整建议提示词
        
        Args:
            current_medication: 当前药物
            current_dosage: 当前剂量
            treatment_response: 治疗反应
            side_effects: 副作用
            
        Returns:
            提示词字符串
        """
        prompt = f"""请评估用药方案并提供调整建议。

【当前用药】
药品：{current_medication}
剂量：{current_dosage}

【治疗反应】
{treatment_response}

"""
        
        if side_effects:
            prompt += f"【副作用】\n{', '.join(side_effects)}\n\n"
        
        prompt += """【评估要点】
1. 当前剂量是否合适？
2. 是否需要调整剂量？
3. 是否需要更换药物？
4. 如何处理副作用？
5. 调整的时机和方法

【调整建议】
请提供具体的调整方案和理由。
"""
        return prompt
    
    @staticmethod
    def get_drug_interaction_check_prompt(
        medications: List[str],
        new_drug: str
    ) -> str:
        """
        生成药物相互作用检查提示词
        
        Args:
            medications: 当前药物列表
            new_drug: 新增药物
            
        Returns:
            提示词字符串
        """
        prompt = f"""请检查药物相互作用。

【当前用药】
{chr(10).join(f'- {med}' for med in medications)}

【新增药物】
{new_drug}

【检查内容】
1. 是否存在药物相互作用？
2. 相互作用的类型和严重程度
3. 可能的临床表现
4. 如何监测和管理
5. 是否需要调整用药方案

请提供详细的相互作用分析。
"""
        return prompt