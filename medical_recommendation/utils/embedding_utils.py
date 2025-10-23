"""
嵌入向量工具模块
提供文本嵌入生成、相似度计算等功能
"""

import numpy as np
from typing import List, Union, Optional, Tuple, Dict, Any
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """嵌入模型提供者抽象基类"""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        生成单个文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文本的嵌入向量
        
        Args:
            texts: 输入文本列表
            
        Returns:
            嵌入向量列表
        """
        pass


class EmbeddingUtils:
    """
    嵌入向量工具类
    提供向量生成、相似度计算、向量操作等功能
    """
    
    def __init__(self, provider: Optional[EmbeddingProvider] = None):
        """
        初始化嵌入向量工具
        
        Args:
            provider: 嵌入模型提供者实例
        """
        self.provider = provider
        self._cache: Dict[str, List[float]] = {}
        self.enable_cache = True
    
    def set_provider(self, provider: EmbeddingProvider) -> None:
        """
        设置嵌入模型提供者
        
        Args:
            provider: 嵌入模型提供者实例
        """
        self.provider = provider
        logger.info(f"嵌入模型提供者已设置: {type(provider).__name__}")
    
    def embed_text(
        self, 
        text: str, 
        use_cache: bool = True
    ) -> List[float]:
        """
        生成单个文本的嵌入向量
        
        Args:
            text: 输入文本
            use_cache: 是否使用缓存
            
        Returns:
            嵌入向量
            
        Raises:
            ValueError: 提供者未设置
        """
        if self.provider is None:
            raise ValueError("嵌入模型提供者未设置")
        
        # 检查缓存
        if use_cache and self.enable_cache and text in self._cache:
             logger.debug(f"从缓存获取嵌入向量")
             return self._cache[text]
        
        try:
            embedding = self.provider.embed_text(text)
            
            # 缓存结果
            if use_cache and self.enable_cache:
                self._cache[text] = embedding
            
            return embedding
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            raise
    
    def embed_texts(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        批量生成文本的嵌入向量
        
        Args:
            texts: 输入文本列表
            batch_size: 批处理大小（可选）
            
        Returns:
            嵌入向量列表
            
        Raises:
            ValueError: 提供者未设置
        """
        if self.provider is None:
            raise ValueError("嵌入模型提供者未设置")
        
        if not texts:
            return []
        
        # 如果指定了batch_size，分批处理
        if batch_size and batch_size > 0:
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.provider.embed_texts(batch)
                all_embeddings.extend(batch_embeddings)
            return all_embeddings
        
        return self.provider.embed_texts(texts)
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            余弦相似度 [-1, 1]
        """
        arr1 = np.array(vec1)
        arr2 = np.array(vec2)
        
        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    @staticmethod
    def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的欧几里得距离
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            欧几里得距离
        """
        arr1 = np.array(vec1)
        arr2 = np.array(vec2)
        return float(np.linalg.norm(arr1 - arr2))
    
    @staticmethod
    def dot_product(vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的点积
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            点积值
        """
        arr1 = np.array(vec1)
        arr2 = np.array(vec2)
        return float(np.dot(arr1, arr2))
    
    @staticmethod
    def normalize_vector(vec: List[float]) -> List[float]:
        """
        归一化向量
        
        Args:
            vec: 输入向量
            
        Returns:
            归一化后的向量
        """
        arr = np.array(vec)
        norm = np.linalg.norm(arr)
        
        if norm == 0:
            return vec
        
        normalized = arr / norm
        return normalized.tolist()
    
    def compute_similarity_matrix(
        self, 
        embeddings: List[List[float]], 
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        计算嵌入向量之间的相似度矩阵
        
        Args:
            embeddings: 嵌入向量列表
            metric: 相似度度量方式 (cosine, euclidean, dot_product)
            
        Returns:
            相似度矩阵
        """
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if metric == "cosine":
                    sim = self.cosine_similarity(embeddings[i], embeddings[j])
                elif metric == "euclidean":
                    # 转换为相似度（距离越小，相似度越高）
                    dist = self.euclidean_distance(embeddings[i], embeddings[j])
                    sim = 1.0 / (1.0 + dist)
                elif metric == "dot_product":
                    sim = self.dot_product(embeddings[i], embeddings[j])
                else:
                    raise ValueError(f"不支持的相似度度量方式: {metric}")
                
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        return similarity_matrix
    
    def find_most_similar(
        self, 
        query_embedding: List[float], 
        candidate_embeddings: List[List[float]], 
        top_k: int = 5,
        metric: str = "cosine"
    ) -> List[Tuple[int, float]]:
        """
        找到最相似的候选向量
        
        Args:
            query_embedding: 查询向量
            candidate_embeddings: 候选向量列表
            top_k: 返回前k个最相似的
            metric: 相似度度量方式
            
        Returns:
            (索引, 相似度分数) 元组列表，按相似度降序排列
        """
        similarities = []
        
        for idx, candidate in enumerate(candidate_embeddings):
            if metric == "cosine":
                sim = self.cosine_similarity(query_embedding, candidate)
            elif metric == "euclidean":
                dist = self.euclidean_distance(query_embedding, candidate)
                sim = 1.0 / (1.0 + dist)
            elif metric == "dot_product":
                sim = self.dot_product(query_embedding, candidate)
            else:
                raise ValueError(f"不支持的相似度度量方式: {metric}")
            
            similarities.append((idx, sim))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    @staticmethod
    def average_embeddings(embeddings: List[List[float]]) -> List[float]:
        """
        计算多个嵌入向量的平均值
        
        Args:
            embeddings: 嵌入向量列表
            
        Returns:
            平均嵌入向量
        """
        if not embeddings:
            return []
        
        arr = np.array(embeddings)
        avg = np.mean(arr, axis=0)
        return avg.tolist()
    
    @staticmethod
    def weighted_average_embeddings(
        embeddings: List[List[float]], 
        weights: List[float]
    ) -> List[float]:
        """
        计算加权平均嵌入向量
        
        Args:
            embeddings: 嵌入向量列表
            weights: 权重列表
            
        Returns:
            加权平均嵌入向量
        """
        if not embeddings or not weights or len(embeddings) != len(weights):
            raise ValueError("嵌入向量和权重列表长度必须相同且不为空")
        
        arr = np.array(embeddings)
        w = np.array(weights)
        
        # 归一化权重
        w = w / np.sum(w)
        
        weighted_avg = np.average(arr, axis=0, weights=w)
        return weighted_avg.tolist()
    
    def clear_cache(self) -> None:
        """清空嵌入向量缓存"""
        self._cache.clear()
        logger.info("嵌入向量缓存已清空")
    
    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)
# 全局工具实例
_global_embedding_utils: Optional[EmbeddingUtils] = None
def get_embedding_utils(provider: Optional[EmbeddingProvider] = None) -> EmbeddingUtils:
    """
    获取全局嵌入向量工具实例（单例模式）
    
    Args:
        provider: 嵌入模型提供者（可选）
        
    Returns:
        EmbeddingUtils实例
    """
    global _global_embedding_utils
    if _global_embedding_utils is None:
        _global_embedding_utils = EmbeddingUtils(provider)
    elif provider is not None:
        _global_embedding_utils.set_provider(provider)
    return _global_embedding_utils
def embed_text(text: str, use_cache: bool = True) -> List[float]:
    """
    便捷函数：生成文本嵌入向量
    
    Args:
        text: 输入文本
        use_cache: 是否使用缓存
        
    Returns:
        嵌入向量
    """
    utils = get_embedding_utils()
    return utils.embed_text(text, use_cache)
def compute_similarity(
    vec1: List[float], 
    vec2: List[float], 
    metric: str = "cosine"
) -> float:
    """
    便捷函数：计算向量相似度
    
    Args:
        vec1: 向量1
        vec2: 向量2
        metric: 相似度度量方式
        
    Returns:
        相似度分数
    """
    if metric == "cosine":
        return EmbeddingUtils.cosine_similarity(vec1, vec2)
    elif metric == "euclidean":
        dist = EmbeddingUtils.euclidean_distance(vec1, vec2)
        return 1.0 / (1.0 + dist)
    elif metric == "dot_product":
        return EmbeddingUtils.dot_product(vec1, vec2)
    else:
        raise ValueError(f"不支持的相似度度量方式: {metric}")