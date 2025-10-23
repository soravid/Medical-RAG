"""
混合检索器
整合知识图谱检索和向量检索
"""

import logging
from typing import List, Dict, Any, Optional
from ..retrievers import EnsembleRetriever, MergeStrategy

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    混合检索器
    整合多种检索策略提供统一接口
    """
    
    def __init__(
        self,
        kg_manager: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        ensemble_retriever: Optional[EnsembleRetriever] = None,
        use_mock: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化混合检索器
        
        Args:
            kg_manager: 知识图谱管理器
            vector_store: 向量存储管理器
            ensemble_retriever: 集成检索器
            use_mock: 是否使用模拟数据
            config: 配置字典
        """
        self.kg_manager = kg_manager
        self.vector_store = vector_store
        self.ensemble_retriever = ensemble_retriever
        self.use_mock = use_mock
        self.config = config or {}
        
        # 默认配置
        self.default_top_k = self.config.get("top_k", 10)
        self.kg_weight = self.config.get("kg_weight", 0.5)
        self.vector_weight = self.config.get("vector_weight", 0.5)
        self.merge_strategy = self.config.get("merge_strategy", MergeStrategy.LINEAR_COMBINATION)
        
        logger.info("混合检索器初始化完成")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: str = "hybrid",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        执行检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            mode: 检索模式 (hybrid/kg/vector)
            **kwargs: 其他参数
            
        Returns:
            检索结果列表
        """
        top_k = top_k or self.default_top_k
        
        if mode == "kg":
            return self._kg_retrieve(query, top_k)
        elif mode == "vector":
            return self._vector_retrieve(query, top_k)
        else:  # hybrid
            return self._hybrid_retrieve(query, top_k, **kwargs)
    
    def _hybrid_retrieve(
        self,
        query: str,
        top_k: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """混合检索"""
        if self.ensemble_retriever and not self.use_mock:
            # 使用集成检索器
            results = self.ensemble_retriever.retrieve(query, top_k=top_k)
            return [
                {
                    "id": r.result_id,
                    "content": r.content,
                    "score": r.final_score,
                    "type": r.result_type,
                    "source_scores": r.source_scores
                }
                for r in results
            ]
        else:
            # 手动合并
            kg_results = self._kg_retrieve(query, top_k)
            vector_results = self._vector_retrieve(query, top_k)
            
            return self._merge_results(kg_results, vector_results, top_k)
    
    def _kg_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """知识图谱检索"""
        if self.use_mock or self.kg_manager is None:
            return self._mock_kg_retrieve(query, top_k)
        
        try:
            # 简单的实体查询
            entities = self.kg_manager.query_entity(
                entity_type="Disease",
                entity_name=None
            )
            
            return [
                {
                    "id": e["id"],
                    "content": str(e["properties"]),
                    "score": 0.8,
                    "type": "kg"
                }
                for e in entities[:top_k]
            ]
        except Exception as e:
            logger.error(f"知识图谱检索失败: {e}")
            return []
    
    def _vector_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """向量检索"""
        if self.use_mock or self.vector_store is None:
            return self._mock_vector_retrieve(query, top_k)
        
        try:
            results = self.vector_store.search(
                query=query,
                top_k=top_k
            )
            
            return [
                {
                    "id": r["doc_id"],
                    "content": r.get("text", ""),
                    "score": r["score"],
                    "type": "vector"
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []
    
    def _merge_results(
        self,
        kg_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """合并结果"""
        # 简单的线性组合
        all_results = []
        
        for r in kg_results:
            r["final_score"] = r["score"] * self.kg_weight
            all_results.append(r)
        
        for r in vector_results:
            r["final_score"] = r["score"] * self.vector_weight
            all_results.append(r)
        
        # 排序并返回
        all_results.sort(key=lambda x: x["final_score"], reverse=True)
        return all_results[:top_k]
    
    def _mock_kg_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """模拟知识图谱检索"""
        return [
            {
                "id": "kg_001",
                "content": "高血压是一种慢性疾病",
                "score": 0.85,
                "type": "kg"
            },
            {
                "id": "kg_002",
                "content": "糖尿病需要长期管理",
                "score": 0.80,
                "type": "kg"
            }
        ][:top_k]
    
    def _mock_vector_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """模拟向量检索"""
        return [
            {
                "id": "vec_001",
                "content": "高血压的治疗方法包括药物和生活方式调整",
                "score": 0.90,
                "type": "vector"
            },
            {
                "id": "vec_002",
                "content": "定期监测血压对高血压患者很重要",
                "score": 0.88,
                "type": "vector"
            }
        ][:top_k]
    
    def retrieve_by_entity(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        基于实体检索
        
        Args:
            entity_id: 实体ID
            relationship_type: 关系类型
            top_k: 返回结果数量
            
        Returns:
            检索结果
        """
        if self.kg_manager:
            relationships = self.kg_manager.query_relationships(
                entity_id,
                relationship_type
            )
            return relationships[:top_k]
        return []