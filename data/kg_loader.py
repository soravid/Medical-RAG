"""
知识图谱数据加载器
提供从各种数据源加载医疗知识图谱数据的功能
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class Entity(BaseModel):
    """实体模型"""
    id: str = Field(..., description="实体唯一标识符")
    label: str = Field(..., description="实体类型标签")
    properties: Dict[str, Any] = Field(default_factory=dict, description="实体属性")


class Relationship(BaseModel):
    """关系模型"""
    id: str = Field(..., description="关系唯一标识符")
    type: str = Field(..., description="关系类型")
    source_id: str = Field(..., description="源实体ID")
    target_id: str = Field(..., description="目标实体ID")
    properties: Dict[str, Any] = Field(default_factory=dict, description="关系属性")


class KGDataSchema(BaseModel):
    """知识图谱数据架构"""
    entities: List[Entity] = Field(default_factory=list, description="实体列表")
    relationships: List[Relationship] = Field(default_factory=list, description="关系列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """根据ID获取实体"""
        for entity in self.entities:
            if entity.id == entity_id:
                return entity
        return None
    
    def get_entities_by_label(self, label: str) -> List[Entity]:
        """根据标签获取实体列表"""
        return [e for e in self.entities if e.label == label]
    
    def get_relationships_by_type(self, rel_type: str) -> List[Relationship]:
        """根据类型获取关系列表"""
        return [r for r in self.relationships if r.type == rel_type]
    
    def get_entity_relationships(
        self, 
        entity_id: str, 
        direction: str = "both"
    ) -> List[Relationship]:
        """
        获取实体的所有关系
        
        Args:
            entity_id: 实体ID
            direction: 方向 (outgoing/incoming/both)
            
        Returns:
            关系列表
        """
        relationships = []
        
        for rel in self.relationships:
            if direction in ["outgoing", "both"] and rel.source_id == entity_id:
                relationships.append(rel)
            elif direction in ["incoming", "both"] and rel.target_id == entity_id:
                relationships.append(rel)
        
        return relationships


class KGLoader:
    """
    知识图谱数据加载器
    支持从JSON、CSV、Neo4j等多种数据源加载数据
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化知识图谱加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir) if data_dir else Path("./data/kg")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_from_json(self, file_path: str) -> KGDataSchema:
        """
        从JSON文件加载知识图谱数据
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            知识图谱数据
        """
        full_path = self.data_dir / file_path if not Path(file_path).is_absolute() else Path(file_path)
        
        if not full_path.exists():
            logger.warning(f"文件不存在: {full_path}，返回模拟数据")
            return self.generate_mock_data()
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            kg_data = KGDataSchema(**data)
            logger.info(f"成功加载知识图谱数据: {len(kg_data.entities)}个实体, {len(kg_data.relationships)}个关系")
            return kg_data
        except Exception as e:
            logger.error(f"加载知识图谱数据失败: {e}")
            raise
    
    def save_to_json(self, kg_data: KGDataSchema, file_path: str) -> None:
        """
        保存知识图谱数据到JSON文件
        
        Args:
            kg_data: 知识图谱数据
            file_path: 保存路径
        """
        full_path = self.data_dir / file_path if not Path(file_path).is_absolute() else Path(file_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(
                    kg_data.model_dump(),
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            logger.info(f"知识图谱数据已保存到: {full_path}")
        except Exception as e:
            logger.error(f"保存知识图谱数据失败: {e}")
            raise
    
    def load_from_neo4j(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j"
    ) -> KGDataSchema:
        """
        从Neo4j数据库加载知识图谱数据
        
        Args:
            uri: Neo4j URI
            username: 用户名
            password: 密码
            database: 数据库名称
            
        Returns:
            知识图谱数据
        """
        # 这里提供接口，实际实现需要安装 neo4j 驱动
        logger.warning("Neo4j加载功能待实现，返回模拟数据")
        return self.generate_mock_data()
    
    def generate_mock_data(self) -> KGDataSchema:
        """
        生成模拟知识图谱数据用于测试
        
        Returns:
            模拟的知识图谱数据
        """
        # 创建疾病实体
        diseases = [
            Entity(
                id="disease_001",
                label="Disease",
                properties={
                    "name": "高血压",
                    "icd_code": "I10",
                    "category": "cardiovascular",
                    "description": "血压持续升高的慢性疾病"
                }
            ),
            Entity(
                id="disease_002",
                label="Disease",
                properties={
                    "name": "糖尿病",
                    "icd_code": "E11",
                    "category": "endocrine",
                    "description": "胰岛素分泌或作用缺陷导致的代谢性疾病"
                }
            ),
            Entity(
                id="disease_003",
                label="Disease",
                properties={
                    "name": "感冒",
                    "icd_code": "J00",
                    "category": "respiratory",
                    "description": "上呼吸道感染"
                }
            ),
        ]
        
        # 创建症状实体
        symptoms = [
            Entity(
                id="symptom_001",
                label="Symptom",
                properties={
                    "name": "头痛",
                    "severity": "moderate",
                    "description": "头部疼痛不适"
                }
            ),
            Entity(
                id="symptom_002",
                label="Symptom",
                properties={
                    "name": "发热",
                    "severity": "moderate",
                    "description": "体温升高"
                }
            ),
            Entity(
                id="symptom_003",
                label="Symptom",
                properties={
                    "name": "口渴",
                    "severity": "mild",
                    "description": "经常感到口渴"
                }
            ),
            Entity(
                id="symptom_004",
                label="Symptom",
                properties={
                    "name": "咳嗽",
                    "severity": "mild",
                    "description": "咳嗽症状"
                }
            ),
        ]
        
        # 创建药品实体
        drugs = [
            Entity(
                id="drug_001",
                label="Drug",
                properties={
                    "name": "阿司匹林",
                    "generic_name": "乙酰水杨酸",
                    "category": "analgesic",
                    "dosage_form": "片剂"
                }
            ),
            Entity(
                id="drug_002",
                label="Drug",
                properties={
                    "name": "二甲双胍",
                    "generic_name": "盐酸二甲双胍",
                    "category": "endocrine",
                    "dosage_form": "片剂"
                }
            ),
            Entity(
                id="drug_003",
                label="Drug",
                properties={
                    "name": "氨氯地平",
                    "generic_name": "氨氯地平",
                    "category": "cardiovascular",
                    "dosage_form": "片剂"
                }
            ),
        ]
        
        # 创建关系
        relationships = [
            # 疾病-症状关系
            Relationship(
                id="rel_001",
                type="HAS_SYMPTOM",
                source_id="disease_001",
                target_id="symptom_001",
                properties={"frequency": "common", "severity": "moderate"}
            ),
            Relationship(
                id="rel_002",
                type="HAS_SYMPTOM",
                source_id="disease_002",
                target_id="symptom_003",
                properties={"frequency": "very_common", "severity": "mild"}
            ),
            Relationship(
                id="rel_003",
                type="HAS_SYMPTOM",
                source_id="disease_003",
                target_id="symptom_002",
                properties={"frequency": "common", "severity": "moderate"}
            ),
            Relationship(
                id="rel_004",
                type="HAS_SYMPTOM",
                source_id="disease_003",
                target_id="symptom_004",
                properties={"frequency": "very_common", "severity": "mild"}
            ),
            # 疾病-药品关系
            Relationship(
                id="rel_005",
                type="TREATED_BY",
                source_id="disease_001",
                target_id="drug_003",
                properties={"effectiveness": "high", "line": "first"}
            ),
            Relationship(
                id="rel_006",
                type="TREATED_BY",
                source_id="disease_002",
                target_id="drug_002",
                properties={"effectiveness": "high", "line": "first"}
            ),
            Relationship(
                id="rel_007",
                type="TREATED_BY",
                source_id="disease_003",
                target_id="drug_001",
                properties={"effectiveness": "moderate", "line": "symptomatic"}
            ),
            # 症状-药品关系
            Relationship(
                id="rel_008",
                type="RELIEVES",
                source_id="drug_001",
                target_id="symptom_001",
                properties={"effectiveness": "high"}
            ),
            Relationship(
                id="rel_009",
                type="RELIEVES",
                source_id="drug_001",
                target_id="symptom_002",
                properties={"effectiveness": "moderate"}
            ),
        ]
        
        # 组合所有实体
        all_entities = diseases + symptoms + drugs
        
        # 创建元数据
        metadata = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "entity_count": len(all_entities),
            "relationship_count": len(relationships),
            "entity_types": {
                "Disease": len(diseases),
                "Symptom": len(symptoms),
                "Drug": len(drugs),
            },
            "relationship_types": {
                "HAS_SYMPTOM": 4,
                "TREATED_BY": 3,
                "RELIEVES": 2,
            }
        }
        
        kg_data = KGDataSchema(
            entities=all_entities,
            relationships=relationships,
            metadata=metadata
        )
        
        logger.info(f"生成模拟数据: {len(all_entities)}个实体, {len(relationships)}个关系")
        return kg_data
    
    def validate_data(self, kg_data: KGDataSchema) -> Tuple[bool, List[str]]:
        """
        验证知识图谱数据完整性
        
        Args:
            kg_data: 知识图谱数据
            
        Returns:
            (是否通过验证, 错误信息列表)
        """
        errors = []
        
        # 检查实体ID唯一性
        entity_ids = [e.id for e in kg_data.entities]
        if len(entity_ids) != len(set(entity_ids)):
            errors.append("存在重复的实体ID")
        
        # 检查关系引用的实体是否存在
        entity_id_set = set(entity_ids)
        for rel in kg_data.relationships:
            if rel.source_id not in entity_id_set:
                errors.append(f"关系 {rel.id} 引用了不存在的源实体: {rel.source_id}")
            if rel.target_id not in entity_id_set:
                errors.append(f"关系 {rel.id} 引用了不存在的目标实体: {rel.target_id}")
        
        is_valid = len(errors) == 0
        if is_valid:
            logger.info("知识图谱数据验证通过")
        else:
            logger.warning(f"知识图谱数据验证失败: {errors}")
        
        return is_valid, errors
    
    def get_statistics(self, kg_data: KGDataSchema) -> Dict[str, Any]:
        """
        获取知识图谱统计信息
        
        Args:
            kg_data: 知识图谱数据
            
        Returns:
            统计信息字典
        """
        entity_labels = {}
        for entity in kg_data.entities:
            entity_labels[entity.label] = entity_labels.get(entity.label, 0) + 1
        
        relationship_types = {}
        for rel in kg_data.relationships:
            relationship_types[rel.type] = relationship_types.get(rel.type, 0) + 1
        
        return {
            "total_entities": len(kg_data.entities),
            "total_relationships": len(kg_data.relationships),
            "entity_types": entity_labels,
            "relationship_types": relationship_types,
        }


def load_kg_data(
    source: str = "mock",
    data_dir: Optional[str] = None,
    **kwargs
) -> KGDataSchema:
    """
    便捷函数：加载知识图谱数据
    
    Args:
        source: 数据源 (mock/json/neo4j)
        data_dir: 数据目录
        **kwargs: 其他参数
        
    Returns:
        知识图谱数据
    """
    loader = KGLoader(data_dir)
    
    if source == "mock":
        return loader.generate_mock_data()
    elif source == "json":
        file_path = kwargs.get("file_path", "kg_data.json")
        return loader.load_from_json(file_path)
    elif source == "neo4j":
        return loader.load_from_neo4j(
            uri=kwargs.get("uri", "bolt://localhost:7687"),
            username=kwargs.get("username", "neo4j"),
            password=kwargs.get("password", "password"),
            database=kwargs.get("database", "neo4j")
        )
    else:
        raise ValueError(f"不支持的数据源: {source}")