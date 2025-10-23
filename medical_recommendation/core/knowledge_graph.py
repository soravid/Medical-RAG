"""
知识图谱管理
封装知识图谱的操作和查询
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class KnowledgeGraphManager:
    """
    知识图谱管理器
    提供统一的知识图谱操作接口
    """
    
    def __init__(self, kg_data: Optional[Any] = None, use_mock: bool = True):
        """
        初始化知识图谱管理器
        
        Args:
            kg_data: 知识图谱数据
            use_mock: 是否使用模拟数据
        """
        self.kg_data = kg_data
        self.use_mock = use_mock
        
        if self.kg_data is None and not use_mock:
            logger.warning("未提供知识图谱数据，将使用模拟数据")
            self.use_mock = True
        
        logger.info("知识图谱管理器初始化完成")
    
    def query_entity(
        self,
        entity_type: str,
        entity_name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        查询实体
        
        Args:
            entity_type: 实体类型
            entity_name: 实体名称
            properties: 属性过滤
            
        Returns:
            实体列表
        """
        if self.use_mock:
            return self._mock_query_entity(entity_type, entity_name)
        
        try:
            entities = self.kg_data.get_entities_by_label(entity_type)
            
            # 应用过滤
            if entity_name:
                entities = [
                    e for e in entities
                    if e.properties.get("name", "").lower() == entity_name.lower()
                ]
            
            if properties:
                filtered = []
                for entity in entities:
                    match = all(
                        entity.properties.get(k) == v
                        for k, v in properties.items()
                    )
                    if match:
                        filtered.append(entity)
                entities = filtered
            
            return [
                {
                    "id": e.id,
                    "label": e.label,
                    "properties": e.properties
                }
                for e in entities
            ]
        
        except Exception as e:
            logger.error(f"查询实体失败: {e}")
            return []
    
    def query_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        查询实体关系
        
        Args:
            entity_id: 实体ID
            relationship_type: 关系类型
            
        Returns:
            关系列表
        """
        if self.use_mock:
            return self._mock_query_relationships(entity_id)
        
        try:
            relationships = self.kg_data.get_entity_relationships(entity_id)
            
            if relationship_type:
                relationships = [
                    r for r in relationships
                    if r.type == relationship_type
                ]
            
            return [
                {
                    "id": r.id,
                    "type": r.type,
                    "source": r.source_id,
                    "target": r.target_id,
                    "properties": r.properties
                }
                for r in relationships
            ]
        
        except Exception as e:
            logger.error(f"查询关系失败: {e}")
            return []
    
    def find_path(
        self,
        start_entity: str,
        end_entity: str,
        max_depth: int = 3
    ) -> List[List[str]]:
        """
        查找实体间的路径
        
        Args:
            start_entity: 起始实体
            end_entity: 终止实体
            max_depth: 最大深度
            
        Returns:
            路径列表
        """
        # 简化实现，实际应使用图算法
        logger.info(f"查找路径: {start_entity} -> {end_entity}")
        return [[start_entity, end_entity]]
    
    def _mock_query_entity(
        self,
        entity_type: str,
        entity_name: Optional[str]
    ) -> List[Dict[str, Any]]:
        """模拟实体查询"""
        mock_entities = {
            "Disease": [
                {"id": "disease_001", "label": "Disease", "properties": {"name": "高血压"}},
                {"id": "disease_002", "label": "Disease", "properties": {"name": "糖尿病"}},
            ],
            "Drug": [
                {"id": "drug_001", "label": "Drug", "properties": {"name": "阿司匹林"}},
                {"id": "drug_002", "label": "Drug", "properties": {"name": "二甲双胍"}},
            ],
            "Symptom": [
                {"id": "symptom_001", "label": "Symptom", "properties": {"name": "头痛"}},
                {"id": "symptom_002", "label": "Symptom", "properties": {"name": "发热"}},
            ]
        }
        
        entities = mock_entities.get(entity_type, [])
        
        if entity_name:
            entities = [
                e for e in entities
                if e["properties"]["name"].lower() == entity_name.lower()
            ]
        
        return entities
    
    def _mock_query_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        """模拟关系查询"""
        return [
            {
                "id": "rel_001",
                "type": "HAS_SYMPTOM",
                "source": entity_id,
                "target": "symptom_001",
                "properties": {}
            }
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识图谱统计信息"""
        if self.use_mock:
            return {
                "total_entities": 100,
                "total_relationships": 200,
                "entity_types": {"Disease": 30, "Drug": 40, "Symptom": 30}
            }
        
        if self.kg_data:
            from ..data import KGLoader
            loader = KGLoader()
            return loader.get_statistics(self.kg_data)
        
        return {}