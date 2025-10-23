"""
LLM链管理
封装大语言模型的调用和链式处理
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class LLMChainManager:
    """
    LLM链管理器
    管理和执行各种LLM处理链
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        use_mock: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化LLM链管理器
        
        Args:
            llm: 大语言模型实例
            use_mock: 是否使用模拟响应
            config: 配置字典
        """
        self.llm = llm
        self.use_mock = use_mock
        self.config = config or {}
        
        if self.llm is None and not use_mock:
            logger.warning("未提供LLM实例，将使用模拟响应")
            self.use_mock = True
        
        logger.info("LLM链管理器初始化完成")
    
    def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        生成LLM响应
        
        Args:
            prompt: 提示词
            context: 上下文
            max_tokens: 最大token数
            temperature: 温度参数
            
        Returns:
            生成的响应
        """
        if self.use_mock or self.llm is None:
            return self._mock_generate(prompt)
        
        try:
            # 构建完整提示
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\n{prompt}"
            
            # 调用LLM
            response = self.llm.generate(
                full_prompt,
                max_tokens=max_tokens or self.config.get("max_tokens", 500),
                temperature=temperature or self.config.get("temperature", 0.7)
            )
            
            return response
        
        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            return "抱歉，生成响应时遇到问题。"
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """
        文本摘要
        
        Args:
            text: 输入文本
            max_length: 最大长度
            
        Returns:
            摘要文本
        """
        if self.use_mock:
            return text[:max_length] + "..."
        
        prompt = f"请总结以下内容（不超过{max_length}字）：\n\n{text}"
        return self.generate_response(prompt)
    
    def extract_information(
        self,
        text: str,
        fields: List[str]
    ) -> Dict[str, Any]:
        """
        信息抽取
        
        Args:
            text: 输入文本
            fields: 要抽取的字段
            
        Returns:
            抽取结果
        """
        if self.use_mock:
            return {field: f"extracted_{field}" for field in fields}
        
        prompt = f"从以下文本中提取 {', '.join(fields)}：\n\n{text}"
        response = self.generate_response(prompt)
        
        # 解析响应（简化实现）
        result = {}
        for field in fields:
            result[field] = f"提取的{field}"
        
        return result
    
    def _mock_generate(self, prompt: str) -> str:
        """模拟生成响应"""
        if "诊断" in prompt:
            return "根据症状分析，可能是急性上呼吸道感染。建议进一步检查确诊。"
        elif "治疗" in prompt:
            return "建议采用药物治疗结合生活方式调整的综合方案。请在医生指导下用药。"
        elif "药品" in prompt:
            return "推荐的药品包括对症治疗药物，具体用药请咨询医生或药师。"
        else:
            return "这是一个模拟的LLM响应。实际使用时会连接真实的大语言模型。"
    
    def format_chain_output(
        self,
        chain_type: str,
        chain_output: Any
    ) -> str:
        """
        格式化链输出
        
        Args:
            chain_type: 链类型
            chain_output: 链输出
            
        Returns:
            格式化后的文本
        """
        if chain_type == "diagnosis":
            return f"诊断结果：{chain_output.primary_diagnosis.disease_name if chain_output.primary_diagnosis else '未确定'}"
        elif chain_type == "drug_recommendation":
            drugs = [d.drug_name for d in chain_output.primary_drugs[:3]]
            return f"推荐药品：{', '.join(drugs)}"
        elif chain_type == "treatment_plan":
            return f"治疗策略：{chain_output.treatment_strategy}"
        else:
            return str(chain_output)