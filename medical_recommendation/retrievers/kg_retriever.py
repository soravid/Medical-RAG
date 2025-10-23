"""
知识图谱检索器
基于知识图谱进行结构化检索
"""

import logging
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """查询类型枚举"""
    ENTITY = "entity"                    # 实体查询
    RELATIONSHIP = "relationship"        # 关系查询
    PATH = "path"                        # 路径查询
    NEIGHBORHOOD = "neighborhood"        # 邻域查询
    PATTERN = "pattern"                  # 模式匹配


class KGQuery(BaseModel):
    """知识图谱查询模型"""
    query_type: QueryType = Field(..., description="查询类型")
    entity_id: Optional[str] = Field(None, description="实体ID")
    entity_label: Optional[str] = Field(None, description="实体标签")
    relationship_type: Optional[str] = Field(None, description="关系类型")
    properties: Dict[str, Any] = Field(default_factory=dict, description="属性过滤条件")
    max_depth: int = Field(default=2, description="最大查询深度")
    limit: int = Field(default=10, description="返回结果数量限制")


class KGRetrievalResult(BaseModel):
    """知识图谱检索结果"""
    entity_id: str = Field(..., description="实体ID")
    entity_label: str = Field(..., description="实体标签")
    properties: Dict[str, Any] = Field(default_factory=dict, description="实体属性")
    score: float = Field(default=1.0, description="相关性分数")
    path: Optional[List[str]] = Field(None, description="查询路径")
    relationships: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="相关关系"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    def __str__(self) -> str:
        return f"KGResult(id={self.entity_id}, label={self.entity_label}, score={self.score:.3f})"


class KGRetriever:
    """
    知识图谱检索器
    提供基于知识图谱的结构化检索功能
    """
    
    def __init__(
        self,
        kg_data: Optional[Any] = None,
        use_mock: bool = True
    ):
        """
        初始化知识图谱检索器
        
        Args:
            kg_data: 知识图谱数据（KGDataSchema实例）
            use_mock: 是否使用模拟数据
        """
        self.kg_data = kg_data
        self.use_mock = use_mock
        
        if self.kg_data is None and not use_mock:
            logger.warning("未提供知识图谱数据，将使用模拟检索")
            self.use_mock = True
    
    def retrieve(
        self,
        query: KGQuery,
        **kwargs
    ) -> List[KGRetrievalResult]:
        """
        执行知识图谱检索
        
        Args:
            query: 知识图谱查询
            **kwargs: 其他参数
            
        Returns:
            检索结果列表
        """
        if self.use_mock or self.kg_data is None:
            return self._mock_retrieve(query)
        
        # 根据查询类型执行不同的检索策略
        if query.query_type == QueryType.ENTITY:
            return self._retrieve_entity(query)
        elif query.query_type == QueryType.RELATIONSHIP:
            return self._retrieve_by_relationship(query)
        elif query.query_type == QueryType.NEIGHBORHOOD:
            return self._retrieve_neighborhood(query)
        elif query.query_type == QueryType.PATH:
            return self._retrieve_path(query)
        else:
            logger.warning(f"不支持的查询类型: {query.query_type}")
            return []
    
    def _retrieve_entity(self, query: KGQuery) -> List[KGRetrievalResult]:
        """
        检索实体
        
        Args:
            query: 查询对象
            
        Returns:
            检索结果列表
        """
        results = []
        
        # 按标签筛选实体
        if query.entity_label:
            entities = self.kg_data.get_entities_by_label(query.entity_label)
        else:
            entities = self.kg_data.entities
        
        # 按属性过滤
        for entity in entities:
            if self._match_properties(entity.properties, query.properties):
                result = KGRetrievalResult(
                    entity_id=entity.id,
                    entity_label=entity.label,
                    properties=entity.properties,
                    score=1.0,
                    metadata={"query_type": "entity"}
                )
                results.append(result)
        
        return results[:query.limit]
    
    def _retrieve_by_relationship(self, query: KGQuery) -> List[KGRetrievalResult]:
        """
        基于关系检索实体
        
        Args:
            query: 查询对象
            
        Returns:
            检索结果列表
        """
        if not query.entity_id:
            logger.warning("关系查询需要提供entity_id")
            return []
        
        results = []
        relationships = self.kg_data.get_entity_relationships(query.entity_id)
        
        # 过滤关系类型
        if query.relationship_type:
            relationships = [
                r for r in relationships 
                if r.type == query.relationship_type
            ]
        
        # 获取相关实体
        for rel in relationships:
            # 获取目标实体
            target_id = rel.target_id if rel.source_id == query.entity_id else rel.source_id
            target_entity = self.kg_data.get_entity_by_id(target_id)
            
            if target_entity:
                result = KGRetrievalResult(
                    entity_id=target_entity.id,
                    entity_label=target_entity.label,
                    properties=target_entity.properties,
                    score=0.9,
                    relationships=[{
                        "type": rel.type,
                        "source": rel.source_id,
                        "target": rel.target_id,
                        "properties": rel.properties
                    }],
                    metadata={"query_type": "relationship"}
                )
                results.append(result)
        
        return results[:query.limit]
    
    def _retrieve_neighborhood(self, query: KGQuery) -> List[KGRetrievalResult]:
        """
        检索邻域实体（多跳查询）
        
        Args:
            query: 查询对象
            
        Returns:
            检索结果列表
        """
        if not query.entity_id:
            logger.warning("邻域查询需要提供entity_id")
            return []
        
        visited = set()
        results = []
        
        # 使用BFS进行多跳查询
        current_level = {query.entity_id}
        
        for depth in range(query.max_depth):
            next_level = set()
            
            for entity_id in current_level:
                if entity_id in visited:
                    continue
                
                visited.add(entity_id)
                relationships = self.kg_data.get_entity_relationships(entity_id)
                
                for rel in relationships:
                    neighbor_id = rel.target_id if rel.source_id == entity_id else rel.source_id
                    
                    if neighbor_id not in visited:
                        next_level.add(neighbor_id)
                        neighbor = self.kg_data.get_entity_by_id(neighbor_id)
                        
                        if neighbor:
                            # 根据深度计算分数
                            score = 1.0 / (depth + 1)
                            
                            result = KGRetrievalResult(
                                entity_id=neighbor.id,
                                entity_label=neighbor.label,
                                properties=neighbor.properties,
                                score=score,
                                path=[query.entity_id, neighbor_id],
                                metadata={
                                    "query_type": "neighborhood",
                                    "depth": depth + 1
                                }
                            )
                            results.append(result)
            
            current_level = next_level
            
            if not current_level:
                break
        
        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:query.limit]
    
    def _retrieve_path(self, query: KGQuery) -> List[KGRetrievalResult]:
        """
        路径查询（查找两个实体之间的路径）
        
        Args:
            query: 查询对象
            
        Returns:
            检索结果列表
        """
        # 这里提供基础接口，实际实现可以使用图算法
        logger.warning("路径查询功能待完整实现")
        return []
    
    @staticmethod
    def _match_properties(
        entity_props: Dict[str, Any],
        query_props: Dict[str, Any]
    ) -> bool:
        """
        匹配实体属性
        
        Args:
            entity_props: 实体属性
            query_props: 查询属性
            
        Returns:
            是否匹配
        """
        if not query_props:
            return True
        
        for key, value in query_props.items():
            if key not in entity_props or entity_props[key] != value:
                return False
        
        return True
    
    def _mock_retrieve(self, query: KGQuery) -> List[KGRetrievalResult]:
        """
        模拟知识图谱检索
        
        Args:
            query: 查询对象
            
        Returns:
            模拟检索结果
        """
        logger.info(f"执行模拟知识图谱检索: {query.query_type}")
        
        # 根据查询类型返回不同的模拟结果
        if query.query_type == QueryType.ENTITY:
            mock_results = self._generate_mock_entity_results(query)
        elif query.query_type == QueryType.RELATIONSHIP:
            mock_results = self._generate_mock_relationship_results(query)
        elif query.query_type == QueryType.NEIGHBORHOOD:
            mock_results = self._generate_mock_neighborhood_results(query)
        else:
            mock_results = self._generate_mock_entity_results(query)
        
        return mock_results[:query.limit]
    
    def _generate_mock_entity_results(self, query: KGQuery) -> List[KGRetrievalResult]:
        """生成模拟实体检索结果"""
        mock_entities = [
            {
                "id": "disease_001",
                "label": "Disease",
                "properties": {
                    "name": "高血压",
                    "category": "cardiovascular",
                    "icd_code": "I10"
                },
                "score": 0.95
            },
            {
                "id": "disease_002",
                "label": "Disease",
                "properties": {
                    "name": "糖尿病",
                    "category": "endocrine",
                    "icd_code": "E11"
                },
                "score": 0.88
            },
            {
                "id": "symptom_001",
                "label": "Symptom",
                "properties": {
                    "name": "头痛",
                    "severity": "moderate"
                },
                "score": 0.82
            },
        ]
        
        results = []
        for entity in mock_entities:
            result = KGRetrievalResult(
                entity_id=entity["id"],
                entity_label=entity["label"],
                properties=entity["properties"],
                score=entity["score"],
                metadata={"source": "mock", "query_type": str(query.query_type)}
            )
            results.append(result)
        
        return results
    
    def _generate_mock_relationship_results(self, query: KGQuery) -> List[KGRetrievalResult]:
        """生成模拟关系检索结果"""
        mock_results = [
            {
                "id": "drug_001",
                "label": "Drug",
                "properties": {
                    "name": "阿司匹林",
                    "category": "analgesic"
                },
                "score": 0.92,
                "relationships": [{
                    "type": "TREATS",
                    "source": "drug_001",
                    "target": "disease_001",
                    "properties": {"effectiveness": "high"}
                }]
            },
            {
                "id": "symptom_002",
                "label": "Symptom",
                "properties": {
                    "name": "发热",
                    "severity": "moderate"
                },
                "score": 0.85,
                "relationships": [{
                    "type": "HAS_SYMPTOM",
                    "source": "disease_003",
                    "target": "symptom_002",
                    "properties": {"frequency": "common"}
                }]
            },
        ]
        
        results = []
        for item in mock_results:
            result = KGRetrievalResult(
                entity_id=item["id"],
                entity_label=item["label"],
                properties=item["properties"],
                score=item["score"],
                relationships=item.get("relationships", []),
                metadata={"source": "mock", "query_type": str(query.query_type)}
            )
            results.append(result)
        
        return results
    
    def _generate_mock_neighborhood_results(self, query: KGQuery) -> List[KGRetrievalResult]:
        """生成模拟邻域检索结果"""
        mock_results = [
            {
                "id": "entity_001",
                "label": "Disease",
                "properties": {"name": "高血压"},
                "score": 1.0,
                "depth": 0
            },
            {
                "id": "entity_002",
                "label": "Symptom",
                "properties": {"name": "头痛"},
                "score": 0.8,
                "depth": 1
            },
            {
                "id": "entity_003",
                "label": "Drug",
                "properties": {"name": "氨氯地平"},
                "score": 0.7,
                "depth": 1
            },
            {
                "id": "entity_004",
                "label": "Drug",
                "properties": {"name": "降压药"},
                "score": 0.5,
                "depth": 2
            },
        ]
        
        results = []
        for item in mock_results:
            result = KGRetrievalResult(
                entity_id=item["id"],
                entity_label=item["label"],
                properties=item["properties"],
                score=item["score"],
                path=[query.entity_id or "start", item["id"]],
                metadata={
                    "source": "mock",
                    "query_type": str(query.query_type),
                    "depth": item["depth"]
                }
            )
            results.append(result)
        
        return results
    
    def retrieve_by_entity_name(
        self,
        entity_name: str,
        entity_label: Optional[str] = None,
        limit: int = 10
    ) -> List[KGRetrievalResult]:
        """
        便捷方法：根据实体名称检索
        
        Args:
            entity_name: 实体名称
            entity_label: 实体标签
            limit: 返回结果数量
            
        Returns:
            检索结果列表
        """
        query = KGQuery(
            query_type=QueryType.ENTITY,
            entity_label=entity_label,
            properties={"name": entity_name},
            limit=limit
        )
        return self.retrieve(query)
    
    def retrieve_related_entities(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        max_depth: int = 1,
        limit: int = 10
    ) -> List[KGRetrievalResult]:
        """
        便捷方法：检索相关实体
        
        Args:
            entity_id: 实体ID
            relationship_type: 关系类型
            max_depth: 最大深度
            limit: 返回结果数量
            
        Returns:
            检索结果列表
        """
        if max_depth == 1:
            query = KGQuery(
                query_type=QueryType.RELATIONSHIP,
                entity_id=entity_id,
                relationship_type=relationship_type,
                limit=limit
            )
        else:
            query = KGQuery(
                query_type=QueryType.NEIGHBORHOOD,
                entity_id=entity_id,
                max_depth=max_depth,
                limit=limit
            )
        
        return self.retrieve(query)