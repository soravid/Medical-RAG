"""
向量存储管理
封装向量数据库的操作
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    向量存储管理器
    提供统一的向量存储操作接口
    """
    
    def __init__(
        self,
        vector_indexer: Optional[Any] = None,
        embedding_generator: Optional[Any] = None,
        use_mock: bool = True
    ):
        """
        初始化向量存储管理器
        
        Args:
            vector_indexer: 向量索引器
            embedding_generator: 嵌入生成器
            use_mock: 是否使用模拟数据
        """
        self.vector_indexer = vector_indexer
        self.embedding_generator = embedding_generator
        self.use_mock = use_mock
        
        if (self.vector_indexer is None or self.embedding_generator is None) and not use_mock:
            logger.warning("未提供向量索引器或嵌入生成器，将使用模拟数据")
            self.use_mock = True
        
        logger.info("向量存储管理器初始化完成")
    
    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        添加文档到向量存储
        
        Args:
            doc_id: 文档ID
            text: 文档文本
            metadata: 元数据
            
        Returns:
            是否成功
        """
        if self.use_mock:
            logger.info(f"模拟添加文档: {doc_id}")
            return True
        
        try:
            # 生成嵌入向量
            embedding = self.embedding_generator.generate_embedding(text)
            
            # 添加到索引
            self.vector_indexer.index.add_vector(
                vector=embedding,
                doc_id=doc_id,
                metadata=metadata or {}
            )
            
            logger.info(f"文档已添加: {doc_id}")
            return True
        
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            score_threshold: 分数阈值
            filter_metadata: 元数据过滤
            
        Returns:
            搜索结果
        """
        if self.use_mock:
            return self._mock_search(query, top_k)
        
        try:
            # 生成查询向量
            query_vector = self.embedding_generator.generate_embedding(query)
            
            # 执行搜索
            results = self.vector_indexer.search(
                query_vector=query_vector,
                top_k=top_k,
                                score_threshold=score_threshold,
                filter_metadata=filter_metadata
            )
            
            return [
                {
                    "doc_id": doc_id,
                    "score": score,
                    "metadata": metadata
                }
                for doc_id, score, metadata in results
            ]
        
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def _mock_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """模拟搜索"""
        query_lower = query.lower()
        
        mock_results = [
            {
                "doc_id": "doc_001",
                "score": 0.92,
                "text": "高血压的治疗包括药物治疗和生活方式调整。",
                "metadata": {"category": "treatment"}
            },
            {
                "doc_id": "doc_002",
                "score": 0.88,
                "text": "阿司匹林用于预防心血管疾病。",
                "metadata": {"category": "drug"}
            },
            {
                "doc_id": "doc_003",
                "score": 0.85,
                "text": "糖尿病患者需要控制饮食和定期监测血糖。",
                "metadata": {"category": "treatment"}
            }
        ]
        
        return mock_results[:top_k]
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        获取文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            文档信息
        """
        if self.use_mock:
            return {
                "doc_id": doc_id,
                "text": "模拟文档内容",
                "metadata": {}
            }
        
        try:
            vector = self.vector_indexer.index.get_vector(doc_id)
            if vector is not None:
                return {
                    "doc_id": doc_id,
                    "vector": vector.tolist(),
                }
            return None
        except Exception as e:
            logger.error(f"获取文档失败: {e}")
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """
        删除文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            是否成功
        """
        if self.use_mock:
            logger.info(f"模拟删除文档: {doc_id}")
            return True
        
        try:
            # 实际实现需要向量索引器支持删除操作
            logger.warning("删除操作待实现")
            return False
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        if self.use_mock:
            return {
                "total_documents": 100,
                "index_size": "10MB",
                "dimension": 384
            }
        
        if self.vector_indexer:
            return self.vector_indexer.get_statistics()
        
        return {}