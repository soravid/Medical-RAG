"""
向量搜索工具
提供语义相似度搜索功能
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VectorSearchInput(BaseModel):
    """向量搜索工具输入"""
    query_text: str = Field(..., description="查询文本", min_length=1)
    top_k: int = Field(default=5, description="返回结果数量", ge=1, le=20)
    score_threshold: float = Field(
        default=0.0,
        description="相似度分数阈值",
        ge=0.0,
        le=1.0
    )
    filter_category: Optional[str] = Field(None, description="过滤类别")
    include_metadata: bool = Field(default=True, description="是否包含元数据")


class VectorSearchOutput(BaseModel):
    """向量搜索工具输出"""
    success: bool = Field(..., description="搜索是否成功")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="搜索结果")
    result_count: int = Field(default=0, description="结果数量")
    query_info: Dict[str, Any] = Field(default_factory=dict, description="查询信息")
    error_message: Optional[str] = Field(None, description="错误信息")


class VectorSearchTool:
    """
    向量搜索工具
    基于语义相似度搜索医疗文档和知识
    """
    
    def __init__(
        self,
        vector_retriever: Optional[Any] = None,
        use_mock: bool = True
    ):
        """
        初始化向量搜索工具
        
        Args:
            vector_retriever: 向量检索器实例
            use_mock: 是否使用模拟数据
        """
        self.vector_retriever = vector_retriever
        self.use_mock = use_mock
        
        if self.vector_retriever is None and not use_mock:
            logger.warning("未提供向量检索器，将使用模拟搜索")
            self.use_mock = True
    
    @property
    def name(self) -> str:
        return "vector_search"
    
    @property
    def description(self) -> str:
        return (
            "基于语义相似度搜索医疗文档和知识。"
            "适用于查找与问题相关的医疗信息、治疗方案、用药建议等。"
            "示例：搜索高血压的治疗方法、查找阿司匹林的副作用等。"
        )
    
    def run(self, input_data: VectorSearchInput) -> VectorSearchOutput:
        """
        执行向量搜索
        
        Args:
            input_data: 搜索输入
            
        Returns:
            搜索输出
        """
        logger.info(f"执行向量搜索: query='{input_data.query_text[:50]}...'")
        
        try:
            if self.use_mock or self.vector_retriever is None:
                return self._mock_search(input_data)
            
            # 实际搜索逻辑
            results = self._execute_search(input_data)
            
            return VectorSearchOutput(
                success=True,
                results=results,
                result_count=len(results),
                query_info={
                    "query": input_data.query_text,
                    "top_k": input_data.top_k,
                    "threshold": input_data.score_threshold
                }
            )
        
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return VectorSearchOutput(
                success=False,
                results=[],
                result_count=0,
                query_info={},
                error_message=str(e)
            )
    
    def _execute_search(self, input_data: VectorSearchInput) -> List[Dict[str, Any]]:
        """执行实际搜索"""
        from ...retrievers import VectorQuery
        
        # 构建查询
        vector_query = VectorQuery(
            query_text=input_data.query_text,
            top_k=input_data.top_k,
            score_threshold=input_data.score_threshold
        )
        
        if input_data.filter_category:
            vector_query.filter_metadata = {"category": input_data.filter_category}
        
        # 执行搜索
        results = self.vector_retriever.retrieve(vector_query)
        
        # 转换为字典格式
        output_results = []
        for r in results:
            result_dict = {
                "doc_id": r.doc_id,
                "text": r.text,
                "score": r.score
            }
            if input_data.include_metadata:
                result_dict["metadata"] = r.metadata
            output_results.append(result_dict)
        
        return output_results
    
    def _mock_search(self, input_data: VectorSearchInput) -> VectorSearchOutput:
        """模拟搜索"""
        logger.info("使用模拟向量搜索")
        
        query_lower = input_data.query_text.lower()
        
        # 根据查询内容返回模拟结果
        mock_results = []
        
        if any(kw in query_lower for kw in ["高血压", "血压", "hypertension"]):
            mock_results = [
                {
                    "doc_id": "doc_001",
                    "text": "高血压是指动脉血压持续升高，收缩压≥140mmHg和/或舒张压≥90mmHg。需要长期药物治疗和生活方式调整。",
                    "score": 0.92,
                    "metadata": {"category": "disease", "source": "medical_kb"}
                },
                {
                    "doc_id": "doc_002",
                    "text": "高血压的治疗包括药物治疗（如ACEI、ARB、钙通道阻滞剂等）和非药物治疗（低盐饮食、适量运动、戒烟限酒）。",
                    "score": 0.88,
                    "metadata": {"category": "treatment", "source": "medical_kb"}
                }
            ]
        elif any(kw in query_lower for kw in ["糖尿病", "血糖", "diabetes"]):
            mock_results = [
                {
                    "doc_id": "doc_003",
                    "text": "糖尿病是一组以高血糖为特征的代谢性疾病。典型症状为多饮、多尿、多食、体重下降。",
                    "score": 0.90,
                    "metadata": {"category": "disease", "source": "medical_kb"}
                },
                {
                    "doc_id": "doc_004",
                    "text": "2型糖尿病的一线用药是二甲双胍，配合饮食控制和运动疗法。血糖控制不佳时可联合其他降糖药。",
                    "score": 0.85,
                    "metadata": {"category": "treatment", "source": "medical_kb"}
                }
            ]
        elif any(kw in query_lower for kw in ["阿司匹林", "aspirin"]):
            mock_results = [
                {
                    "doc_id": "doc_005",
                    "text": "阿司匹林是一种非甾体抗炎药，具有解热、镇痛、抗炎作用。小剂量可用于预防心血管疾病。",
                    "score": 0.94,
                    "metadata": {"category": "drug", "source": "drug_db"}
                },
                {
                    "doc_id": "doc_006",
                    "text": "阿司匹林的常见副作用包括胃肠道不适、恶心。长期使用应注意胃黏膜保护，有出血倾向者慎用。",
                    "score": 0.87,
                    "metadata": {"category": "drug", "source": "drug_db"}
                }
            ]
        elif any(kw in query_lower for kw in ["感冒", "发热", "咳嗽", "cold", "fever"]):
            mock_results = [
                {
                    "doc_id": "doc_007",
                    "text": "感冒通常由病毒引起，主要症状包括发热、咳嗽、流涕、咽痛等。一般7-10天可自愈。",
                    "score": 0.89,
                    "metadata": {"category": "disease", "source": "medical_kb"}
                },
                {
                    "doc_id": "doc_008",
                    "text": "感冒的治疗以对症治疗为主，包括退热药、止咳药等。注意休息、多饮水。",
                    "score": 0.84,
                    "metadata": {"category": "treatment", "source": "medical_kb"}
                }
            ]
        else:
            # 通用医疗知识
            mock_results = [
                {
                    "doc_id": "doc_general_001",
                    "text": "预防疾病的基本措施包括：均衡饮食、适量运动、充足睡眠、定期体检、戒烟限酒。",
                    "score": 0.75,
                    "metadata": {"category": "prevention", "source": "medical_kb"}
                },
                {
                    "doc_id": "doc_general_002",
                    "text": "如出现持续不适或症状加重，应及时就医，获取专业医疗建议。",
                    "score": 0.70,
                    "metadata": {"category": "general", "source": "medical_kb"}
                }
            ]
        
        # 应用分数阈值过滤
        filtered_results = [
            r for r in mock_results 
            if r["score"] >= input_data.score_threshold
        ]
        
        # 应用类别过滤
        if input_data.filter_category:
            filtered_results = [
                r for r in filtered_results
                if r.get("metadata", {}).get("category") == input_data.filter_category
            ]
        
        # 限制结果数量
        final_results = filtered_results[:input_data.top_k]
        
        # 移除元数据（如果不需要）
        if not input_data.include_metadata:
            for r in final_results:
                r.pop("metadata", None)
        
        return VectorSearchOutput(
            success=True,
            results=final_results,
            result_count=len(final_results),
            query_info={
                "query": input_data.query_text,
                "top_k": input_data.top_k,
                "mock": True
            }
        )
    
    def search_disease_info(self, disease_name: str, top_k: int = 5) -> VectorSearchOutput:
        """
        便捷方法：搜索疾病信息
        
        Args:
            disease_name: 疾病名称
            top_k: 返回结果数量
            
        Returns:
            搜索输出
        """
        input_data = VectorSearchInput(
            query_text=f"{disease_name}的症状、病因和治疗方法",
            top_k=top_k,
            filter_category="disease"
        )
        return self.run(input_data)
    
    def search_treatment_options(self, condition: str, top_k: int = 5) -> VectorSearchOutput:
        """
        便捷方法：搜索治疗方案
        
        Args:
            condition: 疾病或症状
            top_k: 返回结果数量
            
        Returns:
            搜索输出
        """
        input_data = VectorSearchInput(
            query_text=f"{condition}的治疗方法和用药建议",
            top_k=top_k,
            filter_category="treatment"
        )
        return self.run(input_data)
    
    def search_drug_info(self, drug_name: str, top_k: int = 5) -> VectorSearchOutput:
        """
        便捷方法：搜索药品信息
        
        Args:
            drug_name: 药品名称
            top_k: 返回结果数量
            
        Returns:
            搜索输出
        """
        input_data = VectorSearchInput(
            query_text=f"{drug_name}的用法、副作用和注意事项",
            top_k=top_k,
            filter_category="drug"
        )
        return self.run(input_data)