"""
文本分块器
提供多种文本分块策略
"""

import re
import logging
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    """分块策略枚举"""
    FIXED_SIZE = "fixed_size"  # 固定大小分块
    SENTENCE = "sentence"      # 按句子分块
    PARAGRAPH = "paragraph"    # 按段落分块
    SEMANTIC = "semantic"      # 语义分块
    SLIDING_WINDOW = "sliding_window"  # 滑动窗口分块


class Chunk(BaseModel):
    """文本块模型"""
    id: str = Field(..., description="文本块唯一标识符")
    text: str = Field(..., description="文本块内容")
    start_index: int = Field(..., description="在原文中的起始位置")
    end_index: int = Field(..., description="在原文中的结束位置")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    def __len__(self) -> int:
        """返回文本块长度"""
        return len(self.text)


class BaseChunker(ABC):
    """文本分块器抽象基类"""
    
    @abstractmethod
    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        """
        分块文本
        
        Args:
            text: 输入文本
            **kwargs: 其他参数
            
        Returns:
            文本块列表
        """
        pass


class FixedSizeChunker(BaseChunker):
    """固定大小分块器"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        初始化固定大小分块器
        
        Args:
            chunk_size: 块大小（字符数）
            overlap: 重叠大小（字符数）
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        """
        按固定大小分块
        
        Args:
            text: 输入文本
            
        Returns:
            文本块列表
        """
        if not text or len(text) == 0:
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunk = Chunk(
                id=f"chunk_{chunk_id}",
                text=chunk_text,
                start_index=start,
                end_index=end,
                metadata={
                    "chunk_size": len(chunk_text),
                    "strategy": "fixed_size"
                }
            )
            chunks.append(chunk)
            
            chunk_id += 1
            start = end - self.overlap
            
            # 如果剩余文本太短，直接结束
            if start >= len(text) - self.overlap:
                break
        
        logger.info(f"固定大小分块完成: {len(chunks)}个块")
        return chunks


class SentenceChunker(BaseChunker):
    """按句子分块器"""
    
    def __init__(self, sentences_per_chunk: int = 3):
        """
        初始化句子分块器
        
        Args:
            sentences_per_chunk: 每个块包含的句子数
        """
        self.sentences_per_chunk = sentences_per_chunk
    
    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        """
        按句子分块
        
        Args:
            text: 输入文本
            
        Returns:
            文本块列表
        """
        if not text or len(text) == 0:
            return []
        
        # 分割句子（支持中英文）
        sentences = self._split_sentences(text)
        
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(sentences), self.sentences_per_chunk):
            sentence_group = sentences[i:i + self.sentences_per_chunk]
            chunk_text = " ".join(sentence_group)
            
            # 计算在原文中的位置
            start_index = text.find(sentence_group[0])
            end_index = start_index + len(chunk_text)
            
            chunk = Chunk(
                id=f"chunk_{chunk_id}",
                text=chunk_text,
                start_index=start_index,
                end_index=end_index,
                metadata={
                    "sentence_count": len(sentence_group),
                    "strategy": "sentence"
                }
            )
            chunks.append(chunk)
            chunk_id += 1
        
        logger.info(f"句子分块完成: {len(chunks)}个块")
        return chunks
    
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """分割句子"""
        # 简单的句子分割（支持中英文）
        sentence_endings = r'[。！？.!?]+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]


class ParagraphChunker(BaseChunker):
    """按段落分块器"""
    
    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        """
        按段落分块
        
        Args:
            text: 输入文本
            
        Returns:
            文本块列表
        """
        if not text or len(text) == 0:
            return []
        
        # 按换行符分割段落
        paragraphs = text.split('\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_pos = 0
        
        for i, paragraph in enumerate(paragraphs):
            start_index = text.find(paragraph, current_pos)
            end_index = start_index + len(paragraph)
            
            chunk = Chunk(
                id=f"chunk_{i}",
                text=paragraph,
                start_index=start_index,
                end_index=end_index,
                metadata={
                    "paragraph_number": i + 1,
                    "strategy": "paragraph"
                }
            )
            chunks.append(chunk)
            current_pos = end_index
        
        logger.info(f"段落分块完成: {len(chunks)}个块")
        return chunks


class SlidingWindowChunker(BaseChunker):
    """滑动窗口分块器"""
    
    def __init__(self, window_size: int = 512, step_size: int = 256):
        """
        初始化滑动窗口分块器
        
        Args:
            window_size: 窗口大小
            step_size: 步长
        """
        self.window_size = window_size
        self.step_size = step_size
    
    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        """
        使用滑动窗口分块
        
        Args:
            text: 输入文本
            
        Returns:
            文本块列表
        """
        if not text or len(text) == 0:
            return []
        
        chunks = []
        chunk_id = 0
        
        for start in range(0, len(text), self.step_size):
            end = min(start + self.window_size, len(text))
            chunk_text = text[start:end]
            
            chunk = Chunk(
                id=f"chunk_{chunk_id}",
                text=chunk_text,
                start_index=start,
                end_index=end,
                metadata={
                    "window_size": self.window_size,
                    "step_size": self.step_size,
                    "strategy": "sliding_window"
                }
            )
            chunks.append(chunk)
            chunk_id += 1
            
            # 如果已经到达文本末尾，停止
            if end >= len(text):
                break
        
        logger.info(f"滑动窗口分块完成: {len(chunks)}个块")
        return chunks


class TextChunker:
    """
    文本分块器主类
    支持多种分块策略
    """
    
    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
        **kwargs
    ):
        """
        初始化文本分块器
        
        Args:
            strategy: 分块策略
            **kwargs: 策略特定参数
        """
        self.strategy = strategy
        self.chunker = self._create_chunker(strategy, **kwargs)
    
    def _create_chunker(
        self, 
        strategy: ChunkingStrategy, 
        **kwargs
    ) -> BaseChunker:
        """
        创建具体的分块器
        
        Args:
            strategy: 分块策略
            **kwargs: 策略参数
            
        Returns:
            分块器实例
        """
        if strategy == ChunkingStrategy.FIXED_SIZE:
            return FixedSizeChunker(
                chunk_size=kwargs.get("chunk_size", 512),
                overlap=kwargs.get("overlap", 50)
            )
        elif strategy == ChunkingStrategy.SENTENCE:
            return SentenceChunker(
                sentences_per_chunk=kwargs.get("sentences_per_chunk", 3)
            )
        elif strategy == ChunkingStrategy.PARAGRAPH:
            return ParagraphChunker()
        elif strategy == ChunkingStrategy.SLIDING_WINDOW:
            return SlidingWindowChunker(
                window_size=kwargs.get("window_size", 512),
                step_size=kwargs.get("step_size", 256)
            )
        else:
            raise ValueError(f"不支持的分块策略: {strategy}")
    
    def chunk_text(self, text: str, doc_id: Optional[str] = None) -> List[Chunk]:
        """
        分块文本
        
        Args:
            text: 输入文本
            doc_id: 文档ID（可选）
            
        Returns:
            文本块列表
        """
        chunks = self.chunker.chunk(text)
        
        # 添加文档ID到元数据
        if doc_id:
            for chunk in chunks:
                chunk.metadata["doc_id"] = doc_id
        
        return chunks
    
    def chunk_documents(
        self, 
        documents: List[Dict[str, str]]
    ) -> Dict[str, List[Chunk]]:
        """
        批量分块文档
        
        Args:
            documents: 文档列表，每个文档包含 'id' 和 'text' 字段
            
        Returns:
            文档ID到文本块列表的映射
        """
        result = {}
        
        for doc in documents:
            doc_id = doc.get("id")
            text = doc.get("text", "")
            
            if not doc_id or not text:
                logger.warning(f"跳过无效文档: {doc_id}")
                continue
            
            chunks = self.chunk_text(text, doc_id)
            result[doc_id] = chunks
        
        total_chunks = sum(len(chunks) for chunks in result.values())
        logger.info(f"批量分块完成: {len(result)}个文档, {total_chunks}个文本块")
        
        return result
    
    def get_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        获取文本块统计信息
        
        Args:
            chunks: 文本块列表
            
        Returns:
            统计信息字典
        """
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "min_length": min(chunk_lengths),
            "max_length": max(chunk_lengths),
            "avg_length": sum(chunk_lengths) / len(chunk_lengths),
            "total_characters": sum(chunk_lengths)
        }