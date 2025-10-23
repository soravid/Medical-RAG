"""
嵌入向量生成器
提供文本嵌入向量生成功能
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel, Field

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
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        获取嵌入向量维度
        
        Returns:
            向量维度
        """
        pass


class MockEmbeddingProvider(EmbeddingProvider):
    """模拟嵌入提供者（用于测试）"""
    
    def __init__(self, dimension: int = 384):
        """
        初始化模拟嵌入提供者
        
        Args:
            dimension: 向量维度
        """
        self.dimension = dimension
    
    def embed_text(self, text: str) -> List[float]:
        """生成模拟嵌入向量"""
        np.random.seed(hash(text) % (2**32))  # 使用文本哈希作为种子，保证相同文本得到相同向量
        embedding = np.random.randn(self.dimension)
        # 归一化
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量生成模拟嵌入向量"""
        return [self.embed_text(text) for text in texts]
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI嵌入提供者"""
    
    def __init__(
        self, 
        api_key: str,
        model: str = "text-embedding-ada-002",
        dimension: int = 1536
    ):
        """
        初始化OpenAI嵌入提供者
        
        Args:
            api_key: OpenAI API密钥
            model: 模型名称
            dimension: 向量维度
        """
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        # 实际实现需要安装 openai 库
        logger.warning("OpenAI嵌入提供者接口已定义，实际调用需要安装openai库")
    
    def embed_text(self, text: str) -> List[float]:
        """生成OpenAI嵌入向量"""
        # TODO: 实现OpenAI API调用
        logger.warning("使用模拟数据代替OpenAI API调用")
        return MockEmbeddingProvider(self.dimension).embed_text(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量生成OpenAI嵌入向量"""
        # TODO: 实现OpenAI API批量调用
        return [self.embed_text(text) for text in texts]
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace嵌入提供者"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        """
        初始化HuggingFace嵌入提供者
        
        Args:
            model_name: 模型名称
            device: 设备 (cpu/cuda)
        """
        self.model_name = model_name
        self.device = device
        # 实际实现需要安装 sentence-transformers 库
        logger.warning("HuggingFace嵌入提供者接口已定义，实际调用需要安装sentence-transformers库")
    
    def embed_text(self, text: str) -> List[float]:
        """生成HuggingFace嵌入向量"""
        # TODO: 实现HuggingFace模型调用
        logger.warning("使用模拟数据代替HuggingFace模型调用")
        return MockEmbeddingProvider(384).embed_text(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量生成HuggingFace嵌入向量"""
        # TODO: 实现HuggingFace模型批量调用
        return [self.embed_text(text) for text in texts]
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        # 不同模型维度不同，这里返回常见维度
        return 384


class EmbeddingConfig(BaseModel):
    """嵌入生成配置"""
    provider: str = Field(default="mock", description="提供者类型")
    batch_size: int = Field(default=32, description="批处理大小")
    show_progress: bool = Field(default=True, description="是否显示进度")
    normalize: bool = Field(default=True, description="是否归一化向量")
    cache_enabled: bool = Field(default=True, description="是否启用缓存")


class EmbeddingGenerator:
    """
    嵌入向量生成器
    管理嵌入向量的生成和缓存
    """
    
    def __init__(
        self,
        provider: EmbeddingProvider,
        config: Optional[EmbeddingConfig] = None
    ):
        """
        初始化嵌入向量生成器
        
        Args:
            provider: 嵌入提供者
            config: 配置
        """
        self.provider = provider
        self.config = config or EmbeddingConfig()
        self._cache: Dict[str, List[float]] = {}
    
    def generate_embedding(
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
        """
        # 检查缓存
        if use_cache and self.config.cache_enabled and text in self._cache:
            logger.debug("从缓存获取嵌入向量")
            return self._cache[text]
        
        # 生成嵌入向量
        embedding = self.provider.embed_text(text)
        
        # 归一化
        if self.config.normalize:
            embedding = self._normalize(embedding)
        
        # 缓存
        if use_cache and self.config.cache_enabled:
            self._cache[text] = embedding
        
        return embedding
    
    def generate_embeddings(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[List[float]]:
        """
        批量生成嵌入向量
        
        Args:
            texts: 输入文本列表
            use_cache: 是否使用缓存
            
        Returns:
            嵌入向量列表
        """
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        # 检查缓存
        for i, text in enumerate(texts):
            if use_cache and self.config.cache_enabled and text in self._cache:
                embeddings.append(self._cache[text])
            else:
                embeddings.append(None)  # 占位
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # 批量生成未缓存的嵌入向量
        if texts_to_embed:
            logger.info(f"生成 {len(texts_to_embed)} 个新的嵌入向量")
            
            # 分批处理
            new_embeddings = []
            for i in range(0, len(texts_to_embed), self.config.batch_size):
                batch = texts_to_embed[i:i + self.config.batch_size]
                batch_embeddings = self.provider.embed_texts(batch)
                
                # 归一化
                if self.config.normalize:
                    batch_embeddings = [self._normalize(emb) for emb in batch_embeddings]
                
                new_embeddings.extend(batch_embeddings)
                
                if self.config.show_progress:
                    logger.info(f"进度: {min(i + self.config.batch_size, len(texts_to_embed))}/{len(texts_to_embed)}")
            
            # 填充结果并缓存
            for idx, emb in zip(indices_to_embed, new_embeddings):
                embeddings[idx] = emb
                if use_cache and self.config.cache_enabled:
                    self._cache[texts_to_embed[indices_to_embed.index(idx)]] = emb
        
        return embeddings
    
    @staticmethod
    def _normalize(embedding: List[float]) -> List[float]:
        """归一化向量"""
        arr = np.array(embedding)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return embedding
        normalized = arr / norm
        return normalized.tolist()
    
    def get_dimension(self) -> int:
        """获取嵌入向量维度"""
        return self.provider.get_dimension()
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        logger.info("嵌入向量缓存已清空")
    
    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "cache_size": self.get_cache_size(),
            "dimension": self.get_dimension(),
            "provider": type(self.provider).__name__,
            "batch_size": self.config.batch_size,
            "normalize": self.config.normalize,
        }