"""
集成检索器
融合多种检索策略的结果
"""

import logging
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
from collections import defaultdict

from .kg_retriever import KGRetriever, KGRetrievalResult, KGQuery
from .vector_retriever import VectorRetriever, VectorRetrievalResult, VectorQuery

logger = logging.getLogger(__name__)


class MergeStrategy(str, Enum):
    """结果合并策略枚举"""
    LINEAR_COMBINATION = "linear_combination"  # 线性组合
    RECIPROCAL_RANK = "reciprocal_rank"       # 倒数排名融合
    MAX_SCORE = "max_score"                    # 最大分数
    WEIGHTED_VOTE = "weighted_vote"            # 加权投票


class EnsembleRetrievalResult(BaseModel):
    """集成检索结果"""
    result_id: str = Field(..., description="结果ID")
    content: str = Field(..., description="内容文本")
    final_score: float = Field(..., description="最终分数")
    source_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="各来源的分数"
    )
    result_type: str = Field(..., description="结果类型 (kg/vector)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    def __str__(self) -> str:
        return f"EnsembleResult(id={self.result_id}, score={self.final_score:.3f}, type={self.result_type})"


class EnsembleConfig(BaseModel):
    """集成检索配置"""
    merge_strategy: MergeStrategy = Field(
        default=MergeStrategy.LINEAR_COMBINATION,
        description="合并策略"
    )
    kg_weight: float = Field(default=0.5, description="知识图谱权重", ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.5, description="向量检索权重", ge=0.0, le=1.0)
    enable_kg: bool = Field(default=True, description="是否启用知识图谱检索")
    enable_vector: bool = Field(default=True, description="是否启用向量检索")
    top_k: int = Field(default=10, description="最终返回结果数量")
    score_threshold: float = Field(default=0.0, description="最终分数阈值")


class EnsembleRetriever:
    """
    集成检索器
    融合知识图谱检索和向量检索的结果
    """
    
    def __init__(
        self,
        kg_retriever: Optional[KGRetriever] = None,
        vector_retriever: Optional[VectorRetriever] = None,
        config: Optional[EnsembleConfig] = None
    ):
        """
        初始化集成检索器
        
        Args:
            kg_retriever: 知识图谱检索器
            vector_retriever: 向量检索器
            config: 配置
        """
        self.kg_retriever = kg_retriever or KGRetriever(use_mock=True)
        self.vector_retriever = vector_retriever or VectorRetriever(use_mock=True)
        self.config = config or EnsembleConfig()
        
        # 验证权重和
        total_weight = self.config.kg_weight + self.config.vector_weight
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"权重和不为1.0: {total_weight}，将自动归一化")
            self.config.kg_weight /= total_weight
            self.config.vector_weight /= total_weight
    
    def retrieve(
        self,
        query_text: str,
        kg_query: Optional[KGQuery] = None,
        vector_query: Optional[VectorQuery] = None,
        **kwargs
    ) -> List[EnsembleRetrievalResult]:
        """
        执行集成检索
        
        Args:
            query_text: 查询文本
            kg_query: 知识图谱查询（可选）
            vector_query: 向量查询（可选）
            **kwargs: 其他参数
            
        Returns:
            集成检索结果列表
        """
        kg_results = []
        vector_results = []
        
        # 知识图谱检索
        if self.config.enable_kg:
            try:
                if kg_query is None:
                    # 创建默认查询
                    from .kg_retriever import QueryType
                    kg_query = KGQuery(
                        query_type=QueryType.ENTITY,
                        limit=self.config.top_k * 2  # 获取更多结果用于合并
                    )
                
                kg_results = self.kg_retriever.retrieve(kg_query)
                logger.info(f"知识图谱检索返回 {len(kg_results)} 个结果")
            except Exception as e:
                logger.error(f"知识图谱检索失败: {e}")
        
        # 向量检索
        if self.config.enable_vector:
            try:
                if vector_query is None:
                    vector_query = VectorQuery(
                        query_text=query_text,
                        top_k=self.config.top_k * 2
                    )
                
                vector_results = self.vector_retriever.retrieve(vector_query)
                logger.info(f"向量检索返回 {len(vector_results)} 个结果")
            except Exception as e:
                logger.error(f"向量检索失败: {e}")
        
        # 合并结果
        merged_results = self._merge_results(kg_results, vector_results)
        
        # 应用分数阈值
        filtered_results = [
            r for r in merged_results 
            if r.final_score >= self.config.score_threshold
        ]
        
        # 返回top_k结果
        final_results = filtered_results[:self.config.top_k]
        
        logger.info(f"集成检索完成: 返回 {len(final_results)} 个结果")
        return final_results
    
    def _merge_results(
        self,
        kg_results: List[KGRetrievalResult],
        vector_results: List[VectorRetrievalResult]
    ) -> List[EnsembleRetrievalResult]:
        """
        合并检索结果
        
        Args:
            kg_results: 知识图谱检索结果
            vector_results: 向量检索结果
            
        Returns:
            合并后的结果列表
        """
        strategy = self.config.merge_strategy
        
        if strategy == MergeStrategy.LINEAR_COMBINATION:
            return self._linear_combination_merge(kg_results, vector_results)
        elif strategy == MergeStrategy.RECIPROCAL_RANK:
            return self._reciprocal_rank_merge(kg_results, vector_results)
        elif strategy == MergeStrategy.MAX_SCORE:
            return self._max_score_merge(kg_results, vector_results)
        elif strategy == MergeStrategy.WEIGHTED_VOTE:
            return self._weighted_vote_merge(kg_results, vector_results)
        else:
            logger.warning(f"未知的合并策略: {strategy}，使用线性组合")
            return self._linear_combination_merge(kg_results, vector_results)
    
    def _linear_combination_merge(
        self,
        kg_results: List[KGRetrievalResult],
        vector_results: List[VectorRetrievalResult]
    ) -> List[EnsembleRetrievalResult]:
        """
        线性组合合并策略
        final_score = kg_weight * kg_score + vector_weight * vector_score
        
        Args:
            kg_results: 知识图谱检索结果
            vector_results: 向量检索结果
            
        Returns:
            合并后的结果
        """
        results_dict = {}
        
        # 处理知识图谱结果
        for kg_result in kg_results:
            result_id = kg_result.entity_id
            content = self._extract_kg_content(kg_result)
            
            results_dict[result_id] = {
                "content": content,
                "kg_score": kg_result.score,
                "vector_score": 0.0,
                "result_type": "kg",
                "metadata": kg_result.metadata
            }
        
        # 处理向量结果
        for vector_result in vector_results:
            result_id = vector_result.doc_id
            
            if result_id in results_dict:
                # 如果已存在，更新向量分数
                results_dict[result_id]["vector_score"] = vector_result.score
                results_dict[result_id]["result_type"] = "hybrid"
            else:
                # 新增结果
                results_dict[result_id] = {
                    "content": vector_result.text,
                    "kg_score": 0.0,
                    "vector_score": vector_result.score,
                    "result_type": "vector",
                    "metadata": vector_result.metadata
                }
        
        # 计算最终分数
        ensemble_results = []
        for result_id, data in results_dict.items():
            final_score = (
                self.config.kg_weight * data["kg_score"] +
                self.config.vector_weight * data["vector_score"]
            )
            
            result = EnsembleRetrievalResult(
                result_id=result_id,
                content=data["content"],
                final_score=final_score,
                source_scores={
                    "kg": data["kg_score"],
                    "vector": data["vector_score"]
                },
                result_type=data["result_type"],
                metadata=data["metadata"]
            )
            ensemble_results.append(result)
        
        # 按最终分数排序
        ensemble_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return ensemble_results
    
    def _reciprocal_rank_merge(
        self,
        kg_results: List[KGRetrievalResult],
        vector_results: List[VectorRetrievalResult]
    ) -> List[EnsembleRetrievalResult]:
        """
        倒数排名融合（Reciprocal Rank Fusion）
        RRF score = sum(1 / (k + rank))，其中k通常为60
        
        Args:
            kg_results: 知识图谱检索结果
            vector_results: 向量检索结果
            
        Returns:
            合并后的结果
        """
        k = 60  # RRF常数
        results_dict = defaultdict(lambda: {
            "content": "",
            "ranks": [],
            "result_type": "",
            "metadata": {}
        })
        
        # 处理知识图谱结果（按排名）
        for rank, kg_result in enumerate(kg_results, start=1):
            result_id = kg_result.entity_id
            content = self._extract_kg_content(kg_result)
            
            results_dict[result_id]["content"] = content
            results_dict[result_id]["ranks"].append(("kg", rank))
            results_dict[result_id]["result_type"] = "kg"
            results_dict[result_id]["metadata"] = kg_result.metadata
        
        # 处理向量结果（按排名）
        for rank, vector_result in enumerate(vector_results, start=1):
            result_id = vector_result.doc_id
            
            if result_id in results_dict:
                results_dict[result_id]["ranks"].append(("vector", rank))
                results_dict[result_id]["result_type"] = "hybrid"
            else:
                results_dict[result_id]["content"] = vector_result.text
                results_dict[result_id]["ranks"].append(("vector", rank))
                results_dict[result_id]["result_type"] = "vector"
                results_dict[result_id]["metadata"] = vector_result.metadata
        
        # 计算RRF分数
        ensemble_results = []
        for result_id, data in results_dict.items():
            rrf_score = 0.0
            source_scores = {"kg": 0.0, "vector": 0.0}
            
            for source, rank in data["ranks"]:
                score = 1.0 / (k + rank)
                rrf_score += score
                source_scores[source] = score
            
            result = EnsembleRetrievalResult(
                result_id=result_id,
                content=data["content"],
                final_score=rrf_score,
                source_scores=source_scores,
                result_type=data["result_type"],
                metadata=data["metadata"]
            )
            ensemble_results.append(result)
        
        # 按RRF分数排序
        ensemble_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return ensemble_results
    
    def _max_score_merge(
        self,
        kg_results: List[KGRetrievalResult],
        vector_results: List[VectorRetrievalResult]
    ) -> List[EnsembleRetrievalResult]:
        """
        最大分数合并策略
        final_score = max(kg_score, vector_score)
        
        Args:
            kg_results: 知识图谱检索结果
            vector_results: 向量检索结果
            
        Returns:
            合并后的结果
        """
        results_dict = {}
        
        # 处理知识图谱结果
        for kg_result in kg_results:
            result_id = kg_result.entity_id
            content = self._extract_kg_content(kg_result)
            
            results_dict[result_id] = {
                "content": content,
                "kg_score": kg_result.score,
                "vector_score": 0.0,
                "result_type": "kg",
                "metadata": kg_result.metadata
            }
        
        # 处理向量结果
        for vector_result in vector_results:
            result_id = vector_result.doc_id
            
            if result_id in results_dict:
                results_dict[result_id]["vector_score"] = vector_result.score
                results_dict[result_id]["result_type"] = "hybrid"
            else:
                results_dict[result_id] = {
                    "content": vector_result.text,
                    "kg_score": 0.0,
                    "vector_score": vector_result.score,
                    "result_type": "vector",
                    "metadata": vector_result.metadata
                }
        
        # 计算最终分数（取最大值）
        ensemble_results = []
        for result_id, data in results_dict.items():
            final_score = max(data["kg_score"], data["vector_score"])
            
            result = EnsembleRetrievalResult(
                result_id=result_id,
                content=data["content"],
                final_score=final_score,
                source_scores={
                    "kg": data["kg_score"],
                    "vector": data["vector_score"]
                },
                result_type=data["result_type"],
                metadata=data["metadata"]
            )
            ensemble_results.append(result)
        
        # 按最终分数排序
        ensemble_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return ensemble_results
    
    def _weighted_vote_merge(
        self,
        kg_results: List[KGRetrievalResult],
        vector_results: List[VectorRetrievalResult]
    ) -> List[EnsembleRetrievalResult]:
        """
        加权投票合并策略
        根据结果出现的次数和权重进行投票
        
        Args:
            kg_results: 知识图谱检索结果
            vector_results: 向量检索结果
            
        Returns:
            合并后的结果
        """
        results_dict = defaultdict(lambda: {
            "content": "",
            "votes": 0.0,
            "kg_score": 0.0,
            "vector_score": 0.0,
            "result_type": "",
            "metadata": {}
        })
        
        # 处理知识图谱结果
        for kg_result in kg_results:
            result_id = kg_result.entity_id
            content = self._extract_kg_content(kg_result)
            
            results_dict[result_id]["content"] = content
            results_dict[result_id]["votes"] += self.config.kg_weight * kg_result.score
            results_dict[result_id]["kg_score"] = kg_result.score
            results_dict[result_id]["result_type"] = "kg"
            results_dict[result_id]["metadata"] = kg_result.metadata
        
        # 处理向量结果
        for vector_result in vector_results:
            result_id = vector_result.doc_id
            
            results_dict[result_id]["votes"] += self.config.vector_weight * vector_result.score
            results_dict[result_id]["vector_score"] = vector_result.score
            
            if result_id in results_dict and results_dict[result_id]["content"]:
                results_dict[result_id]["result_type"] = "hybrid"
            else:
                results_dict[result_id]["content"] = vector_result.text
                results_dict[result_id]["result_type"] = "vector"
                results_dict[result_id]["metadata"] = vector_result.metadata
        
        # 转换为结果列表
        ensemble_results = []
        for result_id, data in results_dict.items():
            result = EnsembleRetrievalResult(
                result_id=result_id,
                content=data["content"],
                final_score=data["votes"],
                source_scores={
                    "kg": data["kg_score"],
                    "vector": data["vector_score"]
                },
                result_type=data["result_type"],
                metadata=data["metadata"]
            )
            ensemble_results.append(result)
        
        # 按投票数排序
        ensemble_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return ensemble_results
    
    @staticmethod
    def _extract_kg_content(kg_result: KGRetrievalResult) -> str:
        """
        从知识图谱结果中提取内容文本
        
        Args:
            kg_result: 知识图谱检索结果
            
        Returns:
            内容文本
        """
        # 提取实体名称和属性
        name = kg_result.properties.get("name", kg_result.entity_id)
        label = kg_result.entity_label
        
        # 构建描述文本
        content_parts = [f"{label}: {name}"]
        
        # 添加其他重要属性
        for key, value in kg_result.properties.items():
            if key != "name" and value:
                content_parts.append(f"{key}={value}")
        
        # 添加关系信息
        if kg_result.relationships:
            rel_info = f"关系数: {len(kg_result.relationships)}"
            content_parts.append(rel_info)
        
        return ", ".join(content_parts)
    
    def filter_by_type(
        self,
        results: List[EnsembleRetrievalResult],
        result_type: str
    ) -> List[EnsembleRetrievalResult]:
        """
        按结果类型过滤
        
        Args:
            results: 检索结果列表
            result_type: 结果类型 (kg/vector/hybrid)
            
        Returns:
            过滤后的结果
        """
        return [r for r in results if r.result_type == result_type]
    
    def get_top_from_each_source(
        self,
        results: List[EnsembleRetrievalResult],
        top_k: int = 5
    ) -> Dict[str, List[EnsembleRetrievalResult]]:
        """
        从每个来源获取top_k结果
        
        Args:
            results: 检索结果列表
            top_k: 每个来源返回的结果数
            
        Returns:
            按来源分组的结果字典
        """
        kg_results = []
        vector_results = []
        hybrid_results = []
        
        for result in results:
            if result.result_type == "kg":
                kg_results.append(result)
            elif result.result_type == "vector":
                vector_results.append(result)
            elif result.result_type == "hybrid":
                hybrid_results.append(result)
        
        return {
            "kg": kg_results[:top_k],
            "vector": vector_results[:top_k],
            "hybrid": hybrid_results[:top_k]
        }
    
    def adjust_weights(self, kg_weight: float, vector_weight: float) -> None:
        """
        动态调整权重
        
        Args:
            kg_weight: 知识图谱权重
            vector_weight: 向量检索权重
        """
        total = kg_weight + vector_weight
        self.config.kg_weight = kg_weight / total
        self.config.vector_weight = vector_weight / total
        
        logger.info(f"权重已更新: kg={self.config.kg_weight:.2f}, vector={self.config.vector_weight:.2f}")
    
    def get_statistics(
        self,
        results: List[EnsembleRetrievalResult]
    ) -> Dict[str, Any]:
        """
        获取检索结果统计信息
        
        Args:
            results: 检索结果列表
            
        Returns:
            统计信息字典
        """
        type_counts = defaultdict(int)
        scores = []
        
        for result in results:
            type_counts[result.result_type] += 1
            scores.append(result.final_score)
        
        stats = {
            "total_results": len(results),
            "type_distribution": dict(type_counts),
            "merge_strategy": self.config.merge_strategy,
            "weights": {
                "kg": self.config.kg_weight,
                "vector": self.config.vector_weight
            }
        }
        
        if scores:
            stats["score_stats"] = {
                "min": min(scores),
                "max": max(scores),
                "avg": sum(scores) / len(scores)
            }
        
        return stats


class HybridQuery(BaseModel):
    """混合查询模型"""
    query_text: str = Field(..., description="查询文本")
    kg_query: Optional[KGQuery] = Field(None, description="知识图谱查询")
    vector_query: Optional[VectorQuery] = Field(None, description="向量查询")
    top_k: int = Field(default=10, description="返回结果数量")
    merge_strategy: MergeStrategy = Field(
        default=MergeStrategy.LINEAR_COMBINATION,
        description="合并策略"
    )


def create_ensemble_retriever(
    use_mock: bool = True,
    kg_weight: float = 0.5,
    vector_weight: float = 0.5,
    merge_strategy: MergeStrategy = MergeStrategy.LINEAR_COMBINATION
) -> EnsembleRetriever:
    """
    便捷函数：创建集成检索器
    
    Args:
        use_mock: 是否使用模拟数据
        kg_weight: 知识图谱权重
        vector_weight: 向量检索权重
        merge_strategy: 合并策略
        
    Returns:
        集成检索器实例
    """
    config = EnsembleConfig(
        merge_strategy=merge_strategy,
        kg_weight=kg_weight,
        vector_weight=vector_weight
    )
    
    kg_retriever = KGRetriever(use_mock=use_mock)
    vector_retriever = VectorRetriever(use_mock=use_mock)
    
    return EnsembleRetriever(
        kg_retriever=kg_retriever,
        vector_retriever=vector_retriever,
        config=config
    )