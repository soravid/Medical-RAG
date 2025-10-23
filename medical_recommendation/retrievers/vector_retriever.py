"""
向量检索器
基于向量相似度进行语义检索
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class VectorQuery(BaseModel):
    """向量查询模型"""
    query_text: str = Field(..., description="查询文本")
    query_vector: Optional[List[float]] = Field(None, description="查询向量")
    top_k: int = Field(default=5, description="返回前k个结果")
    score_threshold: float = Field(default=0.0, description="分数阈值")
    filter_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="元数据过滤条件"
    )


class VectorRetrievalResult(BaseModel):
    """向量检索结果"""
    doc_id: str = Field(..., description="文档ID")
    text: str = Field(..., description="文档文本")
    score: float = Field(..., description="相似度分数")
    vector: Optional[List[float]] = Field(None, description="文档向量")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    def __str__(self) -> str:
        return f"VectorResult(id={self.doc_id}, score={self.score:.3f}, text={self.text[:50]}...)"


class VectorRetriever:
    """
    向量检索器
    基于向量相似度进行语义检索
    """
    
    def __init__(
        self,
        vector_store: Optional[Any] = None,
        embedding_function: Optional[callable] = None,
        use_mock: bool = True
    ):
        """
        初始化向量检索器
        
        Args:
            vector_store: 向量存储（VectorIndexer实例）
                        embedding_function: 嵌入函数
            use_mock: 是否使用模拟数据
        """
        self.vector_store = vector_store
        self.embedding_function = embedding_function
        self.use_mock = use_mock
        
        if self.vector_store is None and not use_mock:
            logger.warning("未提供向量存储，将使用模拟检索")
            self.use_mock = True
    
    def retrieve(
        self,
        query: VectorQuery,
        **kwargs
    ) -> List[VectorRetrievalResult]:
        """
        执行向量检索
        
        Args:
            query: 向量查询
            **kwargs: 其他参数
            
        Returns:
            检索结果列表
        """
        if self.use_mock or self.vector_store is None:
            return self._mock_retrieve(query)
        
        # 获取查询向量
        if query.query_vector is None:
            if self.embedding_function is None:
                raise ValueError("需要提供query_vector或embedding_function")
            query.query_vector = self.embedding_function(query.query_text)
        
        # 执行向量搜索
        results = self.vector_store.search(
            query_vector=query.query_vector,
            top_k=query.top_k,
            score_threshold=query.score_threshold,
            filter_metadata=query.filter_metadata if query.filter_metadata else None
        )
        
        # 转换为VectorRetrievalResult
        retrieval_results = []
        for doc_id, score, metadata in results:
            # 获取文档文本
            text = metadata.get("text", "")
            
            result = VectorRetrievalResult(
                doc_id=doc_id,
                text=text,
                score=score,
                metadata=metadata
            )
            retrieval_results.append(result)
        
        logger.info(f"向量检索完成: 返回 {len(retrieval_results)} 个结果")
        return retrieval_results
    
    def _mock_retrieve(self, query: VectorQuery) -> List[VectorRetrievalResult]:
        """
        模拟向量检索
        
        Args:
            query: 查询对象
            
        Returns:
            模拟检索结果
        """
        logger.info(f"执行模拟向量检索: query='{query.query_text[:50]}...'")
        
        # 生成模拟结果
        mock_documents = [
            {
                "doc_id": "vec_doc_001",
                "text": "阿司匹林是一种非甾体抗炎药，具有解热、镇痛、抗炎作用，常用于治疗发热、头痛等症状。",
                "score": 0.92,
                "metadata": {
                    "category": "drug",
                    "source": "medical_knowledge_base",
                    "timestamp": datetime.now().isoformat()
                }
            },
            {
                "doc_id": "vec_doc_002",
                "text": "高血压是指动脉血压持续升高，收缩压≥140mmHg和/或舒张压≥90mmHg。需要长期药物治疗控制。",
                "score": 0.88,
                "metadata": {
                    "category": "disease",
                    "source": "medical_knowledge_base",
                    "timestamp": datetime.now().isoformat()
                }
            },
            {
                "doc_id": "vec_doc_003",
                "text": "糖尿病患者应该严格控制饮食，定期监测血糖水平，按时服用降糖药物。",
                "score": 0.85,
                "metadata": {
                    "category": "treatment",
                    "source": "medical_knowledge_base",
                    "timestamp": datetime.now().isoformat()
                }
            },
            {
                "doc_id": "vec_doc_004",
                "text": "二甲双胍是治疗2型糖尿病的一线用药，通过减少肝脏葡萄糖生成和提高胰岛素敏感性来降低血糖。",
                "score": 0.82,
                "metadata": {
                    "category": "drug",
                    "source": "medical_knowledge_base",
                    "timestamp": datetime.now().isoformat()
                }
            },
            {
                "doc_id": "vec_doc_005",
                "text": "感冒通常由病毒引起，主要症状包括发热、咳嗽、流涕、咽痛等。一般7-10天可自愈。",
                "score": 0.78,
                "metadata": {
                    "category": "disease",
                    "source": "medical_knowledge_base",
                    "timestamp": datetime.now().isoformat()
                }
            },
        ]
        
        # 应用分数阈值
        filtered_docs = [
            doc for doc in mock_documents 
            if doc["score"] >= query.score_threshold
        ]
        
        # 应用元数据过滤
        if query.filter_metadata:
            filtered_docs = [
                doc for doc in filtered_docs
                if self._match_metadata(doc["metadata"], query.filter_metadata)
            ]
        
        # 转换为结果对象
        results = []
        for doc in filtered_docs[:query.top_k]:
            result = VectorRetrievalResult(
                doc_id=doc["doc_id"],
                text=doc["text"],
                score=doc["score"],
                metadata=doc["metadata"]
            )
            results.append(result)
        
        return results
    
    @staticmethod
    def _match_metadata(
        doc_metadata: Dict[str, Any],
        filter_metadata: Dict[str, Any]
    ) -> bool:
        """
        匹配元数据
        
        Args:
            doc_metadata: 文档元数据
            filter_metadata: 过滤条件
            
        Returns:
            是否匹配
        """
        for key, value in filter_metadata.items():
            if key not in doc_metadata or doc_metadata[key] != value:
                return False
        return True
    
    def compute_similarity(
        self,
        vector1: List[float],
        vector2: List[float],
        metric: str = "cosine"
    ) -> float:
        """
        计算向量相似度
        
        Args:
            vector1: 向量1
            vector2: 向量2
            metric: 相似度度量方式 (cosine/euclidean/dot_product)
            
        Returns:
            相似度分数
        """
        arr1 = np.array(vector1)
        arr2 = np.array(vector2)
        
        if metric == "cosine":
            # 余弦相似度
            dot = np.dot(arr1, arr2)
            norm1 = np.linalg.norm(arr1)
            norm2 = np.linalg.norm(arr2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot / (norm1 * norm2))
        
        elif metric == "euclidean":
            # 欧几里得距离转相似度
            dist = np.linalg.norm(arr1 - arr2)
            return float(1.0 / (1.0 + dist))
        
        elif metric == "dot_product":
            # 点积
            return float(np.dot(arr1, arr2))
        
        else:
            raise ValueError(f"不支持的相似度度量: {metric}")
    
    def batch_compute_similarity(
        self,
        query_vector: List[float],
        document_vectors: List[List[float]],
        metric: str = "cosine"
    ) -> List[float]:
        """
        批量计算相似度
        
        Args:
            query_vector: 查询向量
            document_vectors: 文档向量列表
            metric: 相似度度量方式
            
        Returns:
            相似度分数列表
        """
        similarities = []
        for doc_vector in document_vectors:
            similarity = self.compute_similarity(query_vector, doc_vector, metric)
            similarities.append(similarity)
        
        return similarities
    
    def rerank_results(
        self,
        results: List[VectorRetrievalResult],
        rerank_function: Optional[callable] = None
    ) -> List[VectorRetrievalResult]:
        """
        重排序检索结果
        
        Args:
            results: 检索结果列表
            rerank_function: 自定义重排序函数
            
        Returns:
            重排序后的结果
        """
        if rerank_function:
            return rerank_function(results)
        
        # 默认按分数排序
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def retrieve_with_text(
        self,
        query_text: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[VectorRetrievalResult]:
        """
        便捷方法：使用文本进行检索
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            score_threshold: 分数阈值
            
        Returns:
            检索结果列表
        """
        query = VectorQuery(
            query_text=query_text,
            top_k=top_k,
            score_threshold=score_threshold
        )
        return self.retrieve(query)
    
    def retrieve_with_vector(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[VectorRetrievalResult]:
        """
        便捷方法：使用向量进行检索
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            score_threshold: 分数阈值
            
        Returns:
            检索结果列表
        """
        query = VectorQuery(
            query_text="",  # 占位
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold
        )
        return self.retrieve(query)