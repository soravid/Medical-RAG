"""
向量数据加载器
提供向量数据加载、验证和管理功能
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import numpy as np

logger = logging.getLogger(__name__)


class VectorDocument(BaseModel):
    """向量文档模型"""
    id: str = Field(..., description="文档唯一标识符")
    text: str = Field(..., description="文档文本内容")
    embedding: Optional[List[float]] = Field(None, description="文档嵌入向量")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")
    
    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """验证文本不为空"""
        if not v or not v.strip():
            raise ValueError("文档文本不能为空")
        return v.strip()
    
    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """验证嵌入向量"""
        if v is not None and len(v) == 0:
            raise ValueError("嵌入向量不能为空列表")
        return v
    
    def has_embedding(self) -> bool:
        """检查是否有嵌入向量"""
        return self.embedding is not None and len(self.embedding) > 0
    
    def get_embedding_dimension(self) -> Optional[int]:
        """获取嵌入向量维度"""
        if self.has_embedding():
            return len(self.embedding)
        return None


class VectorCollection(BaseModel):
    """向量集合模型"""
    name: str = Field(..., description="集合名称")
    documents: List[VectorDocument] = Field(default_factory=list, description="文档列表")
    dimension: Optional[int] = Field(None, description="向量维度")
    distance_metric: str = Field(default="cosine", description="距离度量方式")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="集合元数据")
    
    def add_document(self, document: VectorDocument) -> None:
        """添加文档"""
        # 验证向量维度一致性
        if self.dimension is not None and document.has_embedding():
            if document.get_embedding_dimension() != self.dimension:
                raise ValueError(
                    f"文档向量维度 {document.get_embedding_dimension()} "
                    f"与集合维度 {self.dimension} 不匹配"
                )
        elif self.dimension is None and document.has_embedding():
            self.dimension = document.get_embedding_dimension()
        
        self.documents.append(document)
    
    def get_document_by_id(self, doc_id: str) -> Optional[VectorDocument]:
        """根据ID获取文档"""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None
    
    def get_documents_with_embeddings(self) -> List[VectorDocument]:
        """获取所有有嵌入向量的文档"""
        return [doc for doc in self.documents if doc.has_embedding()]


class VectorLoader:
    """
    向量数据加载器
    支持从JSON、NPY等格式加载向量数据
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化向量数据加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir) if data_dir else Path("./data/vectors")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_from_json(self, file_path: str) -> VectorCollection:
        """
        从JSON文件加载向量数据
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            向量集合
        """
        full_path = self.data_dir / file_path if not Path(file_path).is_absolute() else Path(file_path)
        
        if not full_path.exists():
            logger.warning(f"文件不存在: {full_path}，返回模拟数据")
            return self.generate_mock_data()
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            collection = VectorCollection(**data)
            logger.info(f"成功加载向量数据: {len(collection.documents)}个文档")
            return collection
        except Exception as e:
            logger.error(f"加载向量数据失败: {e}")
            raise
    
    def save_to_json(self, collection: VectorCollection, file_path: str) -> None:
        """
        保存向量数据到JSON文件
        
        Args:
            collection: 向量集合
            file_path: 保存路径
        """
        full_path = self.data_dir / file_path if not Path(file_path).is_absolute() else Path(file_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(
                    collection.model_dump(),
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            logger.info(f"向量数据已保存到: {full_path}")
        except Exception as e:
            logger.error(f"保存向量数据失败: {e}")
            raise
    
    def load_embeddings_from_npy(
        self, 
        file_path: str,
        doc_ids: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """
        从NPY文件加载嵌入向量
        
        Args:
            file_path: NPY文件路径
            doc_ids: 文档ID列表（可选）
            
        Returns:
            文档ID到向量的映射
        """
        full_path = self.data_dir / file_path if not Path(file_path).is_absolute() else Path(file_path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"文件不存在: {full_path}")
        
        try:
            embeddings = np.load(full_path)
            
            if doc_ids is None:
                doc_ids = [f"doc_{i}" for i in range(len(embeddings))]
            
            if len(embeddings) != len(doc_ids):
                raise ValueError(
                    f"嵌入向量数量 {len(embeddings)} 与文档ID数量 {len(doc_ids)} 不匹配"
                )
            
            embedding_dict = {
                doc_id: embedding.tolist()
                for doc_id, embedding in zip(doc_ids, embeddings)
            }
            
            logger.info(f"成功加载 {len(embedding_dict)} 个嵌入向量")
            return embedding_dict
        except Exception as e:
            logger.error(f"加载嵌入向量失败: {e}")
            raise
    
    def save_embeddings_to_npy(
        self, 
        embeddings: Union[List[List[float]], np.ndarray],
        file_path: str
    ) -> None:
        """
        保存嵌入向量到NPY文件
        
        Args:
            embeddings: 嵌入向量列表或数组
            file_path: 保存路径
        """
        full_path = self.data_dir / file_path if not Path(file_path).is_absolute() else Path(file_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            
            np.save(full_path, embeddings)
            logger.info(f"嵌入向量已保存到: {full_path}")
        except Exception as e:
            logger.error(f"保存嵌入向量失败: {e}")
            raise
    
    def generate_mock_data(self, num_docs: int = 10) -> VectorCollection:
        """
        生成模拟向量数据用于测试
        
        Args:
            num_docs: 生成的文档数量
            
        Returns:
            模拟的向量集合
        """
        mock_texts = [
            "阿司匹林是一种常用的解热镇痛药，可用于治疗头痛、发热等症状。",
            "高血压是指动脉血压持续升高的慢性疾病，需要长期用药控制。",
            "糖尿病患者应该控制饮食，定期监测血糖水平。",
            "感冒通常由病毒引起，主要症状包括发热、咳嗽、流涕等。",
            "二甲双胍是治疗2型糖尿病的一线用药，能够降低血糖水平。",
            "心绞痛发作时可以舌下含服硝酸甘油快速缓解症状。",
            "抗生素应在医生指导下使用，不能随意停药或改变剂量。",
            "慢性阻塞性肺病患者需要戒烟并进行呼吸功能锻炼。",
            "布洛芬具有解热、镇痛、抗炎作用，适用于多种疼痛症状。",
            "高血脂患者应该低脂饮食，必要时服用降脂药物。",
        ]
        
        documents = []
        dimension = 384  # 模拟向量维度
        
        for i in range(min(num_docs, len(mock_texts))):
            # 生成随机向量
            embedding = np.random.randn(dimension).tolist()
            
            doc = VectorDocument(
                id=f"doc_{i+1:03d}",
                text=mock_texts[i],
                embedding=embedding,
                metadata={
                    "source": "mock_data",
                    "category": "medical_knowledge",
                    "created_at": datetime.now().isoformat()
                }
            )
            documents.append(doc)
        
        collection = VectorCollection(
            name="medical_documents",
            documents=documents,
            dimension=dimension,
            distance_metric="cosine",
            metadata={
                "created_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "document_count": len(documents)
            }
        )
        
        logger.info(f"生成模拟数据: {len(documents)}个文档, 维度={dimension}")
        return collection
    
    def validate_collection(
        self, 
        collection: VectorCollection
    ) -> tuple[bool, List[str]]:
        """
        验证向量集合数据
        
        Args:
            collection: 向量集合
            
        Returns:
            (是否通过验证, 错误信息列表)
        """
        errors = []
        
        # 检查文档ID唯一性
        doc_ids = [doc.id for doc in collection.documents]
        if len(doc_ids) != len(set(doc_ids)):
            errors.append("存在重复的文档ID")
        
        # 检查向量维度一致性
        embeddings_docs = collection.get_documents_with_embeddings()
        if embeddings_docs:
            dimensions = [doc.get_embedding_dimension() for doc in embeddings_docs]
            if len(set(dimensions)) > 1:
                errors.append(f"文档向量维度不一致: {set(dimensions)}")
            
            if collection.dimension and dimensions[0] != collection.dimension:
                errors.append(
                    f"文档向量维度 {dimensions[0]} 与集合维度 {collection.dimension} 不匹配"
                )
        
        # 检查文本内容
        for doc in collection.documents:
            if not doc.text or len(doc.text.strip()) == 0:
                errors.append(f"文档 {doc.id} 的文本内容为空")
        
        is_valid = len(errors) == 0
        if is_valid:
            logger.info("向量集合数据验证通过")
        else:
            logger.warning(f"向量集合数据验证失败: {errors}")
        
        return is_valid, errors
    
    def get_statistics(self, collection: VectorCollection) -> Dict[str, Any]:
        """
        获取向量集合统计信息
        
        Args:
            collection: 向量集合
            
        Returns:
            统计信息字典
        """
        docs_with_embeddings = collection.get_documents_with_embeddings()
        
        stats = {
            "total_documents": len(collection.documents),
            "documents_with_embeddings": len(docs_with_embeddings),
            "documents_without_embeddings": len(collection.documents) - len(docs_with_embeddings),
            "vector_dimension": collection.dimension,
            "distance_metric": collection.distance_metric,
        }
        
        # 计算文本统计
        text_lengths = [len(doc.text) for doc in collection.documents]
        if text_lengths:
            stats["text_stats"] = {
                "min_length": min(text_lengths),
                "max_length": max(text_lengths),
                "avg_length": sum(text_lengths) / len(text_lengths)
            }
        
        return stats


def load_vector_data(
    source: str = "mock",
    data_dir: Optional[str] = None,
    **kwargs
) -> VectorCollection:
    """
    便捷函数：加载向量数据
    
    Args:
        source: 数据源 (mock/json)
        data_dir: 数据目录
        **kwargs: 其他参数
        
    Returns:
        向量集合
    """
    loader = VectorLoader(data_dir)
    
    if source == "mock":
        num_docs = kwargs.get("num_docs", 10)
        return loader.generate_mock_data(num_docs)
    elif source == "json":
        file_path = kwargs.get("file_path", "vector_data.json")
        return loader.load_from_json(file_path)
    else:
        raise ValueError(f"不支持的数据源: {source}")