"""
向量索引器
提供向量索引构建和管理功能
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import numpy as np
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)


class IndexType(str, Enum):
    """索引类型枚举"""
    FLAT = "flat"              # 平面索引（暴力搜索）
    HNSW = "hnsw"              # 分层可导航小世界图
    IVF = "ivf"                # 倒排文件索引
    LSH = "lsh"                # 局部敏感哈希


class DistanceMetric(str, Enum):
    """距离度量枚举"""
    COSINE = "cosine"          # 余弦相似度
    EUCLIDEAN = "euclidean"    # 欧几里得距离
    DOT_PRODUCT = "dot_product"  # 点积


class IndexConfig(BaseModel):
    """索引配置"""
    index_type: IndexType = Field(default=IndexType.FLAT, description="索引类型")
    distance_metric: DistanceMetric = Field(
        default=DistanceMetric.COSINE,
        description="距离度量"
    )
    dimension: int = Field(..., description="向量维度")
    index_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="索引参数"
    )


class VectorIndex:
    """向量索引基类"""
    
    def __init__(self, config: IndexConfig):
        """
        初始化向量索引
        
        Args:
            config: 索引配置
        """
        self.config = config
        self.vectors: List[np.ndarray] = []
        self.ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
    
    def add_vector(
        self,
        vector: List[float],
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        添加向量到索引
        
        Args:
            vector: 向量
            doc_id: 文档ID
            metadata: 元数据
        """
        if len(vector) != self.config.dimension:
            raise ValueError(
                f"向量维度 {len(vector)} 与索引维度 {self.config.dimension} 不匹配"
            )
        
        self.vectors.append(np.array(vector))
        self.ids.append(doc_id)
        self.metadata.append(metadata or {})
    
    def add_vectors(
        self,
        vectors: List[List[float]],
        doc_ids: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        批量添加向量
        
        Args:
            vectors: 向量列表
            doc_ids: 文档ID列表
            metadata_list: 元数据列表
        """
        if len(vectors) != len(doc_ids):
            raise ValueError("向量数量与文档ID数量不匹配")
        
        if metadata_list is None:
            metadata_list = [{} for _ in range(len(vectors))]
        
        for vector, doc_id, metadata in zip(vectors, doc_ids, metadata_list):
            self.add_vector(vector, doc_id, metadata)
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_func: Optional[callable] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        搜索最相似的向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回前k个结果
            filter_func: 过滤函数
            
        Returns:
            (文档ID, 相似度分数, 元数据) 元组列表
        """
        if not self.vectors:
            return []
        
        query = np.array(query_vector)
        
        # 计算相似度
        similarities = []
        for i, vec in enumerate(self.vectors):
            # 应用过滤器
            if filter_func and not filter_func(self.metadata[i]):
                continue
            
            similarity = self._compute_similarity(query, vec)
            similarities.append((i, similarity))
        
        # 排序并返回top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in similarities[:top_k]:
            results.append((self.ids[idx], score, self.metadata[idx]))
        
        return results
    
    def _compute_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """
        计算向量相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            相似度分数
        """
        if self.config.distance_metric == DistanceMetric.COSINE:
            # 余弦相似度
            dot = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(dot / (norm1 * norm2))
        
        elif self.config.distance_metric == DistanceMetric.EUCLIDEAN:
            # 欧几里得距离转相似度
            dist = np.linalg.norm(vec1 - vec2)
            return float(1.0 / (1.0 + dist))
        
        elif self.config.distance_metric == DistanceMetric.DOT_PRODUCT:
            # 点积
            return float(np.dot(vec1, vec2))
        
        else:
            raise ValueError(f"不支持的距离度量: {self.config.distance_metric}")
    
    def size(self) -> int:
        """返回索引中的向量数量"""
        return len(self.vectors)
    
    def get_vector(self, doc_id: str) -> Optional[np.ndarray]:
        """根据文档ID获取向量"""
        try:
            idx = self.ids.index(doc_id)
            return self.vectors[idx]
        except ValueError:
            return None


class VectorIndexer:
    """
    向量索引器
    管理向量索引的创建、保存和加载
    """
    
    def __init__(self, config: IndexConfig):
        """
        初始化向量索引器
        
        Args:
            config: 索引配置
        """
        self.config = config
        self.index = VectorIndex(config)
    
    def build_index(
        self,
        vectors: List[List[float]],
           doc_ids: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        构建向量索引
        
        Args:
            vectors: 向量列表
            doc_ids: 文档ID列表
            metadata_list: 元数据列表
        """
        logger.info(f"开始构建向量索引: {len(vectors)} 个向量")
        
        self.index.add_vectors(vectors, doc_ids, metadata_list)
        
        logger.info(f"索引构建完成: {self.index.size()} 个向量")
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        搜索相似向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            score_threshold: 相似度分数阈值
            filter_metadata: 元数据过滤条件
            
        Returns:
            搜索结果列表
        """
        # 构建过滤函数
        def filter_func(metadata: Dict[str, Any]) -> bool:
            if filter_metadata is None:
                return True
            for key, value in filter_metadata.items():
                if metadata.get(key) != value:
                    return False
            return True
        
        # 执行搜索
        results = self.index.search(query_vector, top_k, filter_func)
        
        # 应用分数阈值
        if score_threshold is not None:
            results = [r for r in results if r[1] >= score_threshold]
        
        logger.info(f"搜索完成: 返回 {len(results)} 个结果")
        return results
    
    def save(self, save_path: str) -> None:
        """
        保存索引到文件
        
        Args:
            save_path: 保存路径
        """
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存索引数据
        index_data = {
            "config": self.config.model_dump(),
            "ids": self.index.ids,
            "metadata": self.index.metadata,
            "vectors": [vec.tolist() for vec in self.index.vectors]
        }
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
            logger.info(f"索引已保存到: {path}")
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            raise
    
    def load(self, load_path: str) -> None:
        """
        从文件加载索引
        
        Args:
            load_path: 加载路径
        """
        path = Path(load_path)
        
        if not path.exists():
            raise FileNotFoundError(f"索引文件不存在: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # 重建配置
            self.config = IndexConfig(**index_data["config"])
            self.index = VectorIndex(self.config)
            
            # 恢复数据
            self.index.ids = index_data["ids"]
            self.index.metadata = index_data["metadata"]
            self.index.vectors = [np.array(vec) for vec in index_data["vectors"]]
            
            logger.info(f"索引已加载: {self.index.size()} 个向量")
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取索引统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "index_type": self.config.index_type,
            "distance_metric": self.config.distance_metric,
            "dimension": self.config.dimension,
            "total_vectors": self.index.size(),
            "index_params": self.config.index_params
        }
    
    def clear(self) -> None:
        """清空索引"""
        self.index = VectorIndex(self.config)
        logger.info("索引已清空")


class IndexBuilder:
    """
    索引构建器
    提供完整的索引构建流程
    """
    
    def __init__(
        self,
        dimension: int,
        index_type: IndexType = IndexType.FLAT,
        distance_metric: DistanceMetric = DistanceMetric.COSINE
    ):
        """
        初始化索引构建器
        
        Args:
            dimension: 向量维度
            index_type: 索引类型
            distance_metric: 距离度量
        """
        self.config = IndexConfig(
            index_type=index_type,
            distance_metric=distance_metric,
            dimension=dimension
        )
        self.indexer = VectorIndexer(self.config)
    
    def build_from_documents(
        self,
        documents: List[Dict[str, Any]],
        embedding_field: str = "embedding",
        id_field: str = "id"
    ) -> VectorIndexer:
        """
        从文档构建索引
        
        Args:
            documents: 文档列表
            embedding_field: 嵌入向量字段名
            id_field: ID字段名
            
        Returns:
            构建好的索引器
        """
        vectors = []
        doc_ids = []
        metadata_list = []
        
        for doc in documents:
            if embedding_field not in doc or id_field not in doc:
                logger.warning(f"跳过不完整的文档: {doc.get(id_field, 'unknown')}")
                continue
            
            vectors.append(doc[embedding_field])
            doc_ids.append(doc[id_field])
            
            # 提取元数据（排除embedding和id字段）
            metadata = {k: v for k, v in doc.items() if k not in [embedding_field, id_field]}
            metadata_list.append(metadata)
        
        self.indexer.build_index(vectors, doc_ids, metadata_list)
        
        return self.indexer
    
    def save_index(self, save_path: str) -> None:
        """保存索引"""
        self.indexer.save(save_path)
    
    def load_index(self, load_path: str) -> VectorIndexer:
        """加载索引"""
        self.indexer.load(load_path)
        return self.indexer