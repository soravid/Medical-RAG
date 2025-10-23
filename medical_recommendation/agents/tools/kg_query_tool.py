"""
知识图谱查询工具
提供结构化的医疗知识查询功能
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class KGQueryInput(BaseModel):
    """知识图谱查询工具输入"""
    query_type: str = Field(..., description="查询类型 (entity/relationship/path)")
    entity_name: Optional[str] = Field(None, description="实体名称")
    entity_type: Optional[str] = Field(None, description="实体类型")
    relationship_type: Optional[str] = Field(None, description="关系类型")
    source_entity: Optional[str] = Field(None, description="源实体")
    target_entity: Optional[str] = Field(None, description="目标实体")
    properties: Dict[str, Any] = Field(default_factory=dict, description="属性过滤")
    max_results: int = Field(default=10, description="最大返回结果数")


class KGQueryOutput(BaseModel):
    """知识图谱查询工具输出"""
    success: bool = Field(..., description="查询是否成功")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="查询结果")
    result_count: int = Field(default=0, description="结果数量")
    query_info: Dict[str, Any] = Field(default_factory=dict, description="查询信息")
    error_message: Optional[str] = Field(None, description="错误信息")


class BaseTool(ABC):
    """工具抽象基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """执行工具"""
        pass


class KGQueryTool(BaseTool):
    """
    知识图谱查询工具
    支持实体查询、关系查询、路径查询等
    """
    
    def __init__(self, kg_retriever: Optional[Any] = None, use_mock: bool = True):
        """
        初始化知识图谱查询工具
        
        Args:
            kg_retriever: 知识图谱检索器实例
            use_mock: 是否使用模拟数据
        """
        self.kg_retriever = kg_retriever
        self.use_mock = use_mock
        
        if self.kg_retriever is None and not use_mock:
            logger.warning("未提供知识图谱检索器，将使用模拟查询")
            self.use_mock = True
    
    @property
    def name(self) -> str:
        return "knowledge_graph_query"
    
    @property
    def description(self) -> str:
        return (
            "查询医疗知识图谱，获取疾病、症状、药品等实体信息及其关系。"
            "支持查询类型：entity（实体查询）、relationship（关系查询）、path（路径查询）。"
            "示例：查询高血压的症状、查询阿司匹林治疗的疾病等。"
        )
    
    def run(self, input_data: KGQueryInput) -> KGQueryOutput:
        """
        执行知识图谱查询
        
        Args:
            input_data: 查询输入
            
        Returns:
            查询输出
        """
        logger.info(f"执行知识图谱查询: type={input_data.query_type}")
        
        try:
            if self.use_mock or self.kg_retriever is None:
                return self._mock_query(input_data)
            
            # 实际查询逻辑
            results = self._execute_query(input_data)
            
            return KGQueryOutput(
                success=True,
                results=results,
                result_count=len(results),
                query_info={
                    "query_type": input_data.query_type,
                    "entity_name": input_data.entity_name,
                    "entity_type": input_data.entity_type
                }
            )
        
        except Exception as e:
            logger.error(f"知识图谱查询失败: {e}")
            return KGQueryOutput(
                success=False,
                results=[],
                result_count=0,
                query_info={},
                error_message=str(e)
            )
    
    def _execute_query(self, input_data: KGQueryInput) -> List[Dict[str, Any]]:
        """执行实际查询"""
        from ...retrievers import KGQuery, QueryType
        
        # 构建查询
        if input_data.query_type == "entity":
            kg_query = KGQuery(
                query_type=QueryType.ENTITY,
                entity_label=input_data.entity_type,
                properties=input_data.properties,
                limit=input_data.max_results
            )
        elif input_data.query_type == "relationship":
            kg_query = KGQuery(
                query_type=QueryType.RELATIONSHIP,
                entity_id=input_data.source_entity or "",
                relationship_type=input_data.relationship_type,
                limit=input_data.max_results
            )
        else:
            raise ValueError(f"不支持的查询类型: {input_data.query_type}")
        
        # 执行查询
        results = self.kg_retriever.retrieve(kg_query)
        
        # 转换为字典格式
        return [
            {
                "entity_id": r.entity_id,
                "entity_label": r.entity_label,
                "properties": r.properties,
                "score": r.score,
                "relationships": r.relationships
            }
            for r in results
        ]
    
    def _mock_query(self, input_data: KGQueryInput) -> KGQueryOutput:
        """模拟查询"""
        logger.info("使用模拟知识图谱查询")
        
        mock_results = []
        
        # 实体查询
        if input_data.query_type == "entity":
            if input_data.entity_type == "Disease" or "疾病" in str(input_data.entity_name):
                mock_results = [
                    {
                        "entity_id": "disease_001",
                        "entity_label": "Disease",
                        "properties": {
                            "name": "高血压",
                            "category": "cardiovascular",
                            "icd_code": "I10",
                            "description": "血压持续升高的慢性疾病"
                        },
                        "score": 0.95
                    },
                    {
                        "entity_id": "disease_002",
                        "entity_label": "Disease",
                        "properties": {
                            "name": "糖尿病",
                            "category": "endocrine",
                            "icd_code": "E11",
                            "description": "胰岛素分泌或作用缺陷导致的代谢性疾病"
                        },
                        "score": 0.90
                    }
                ]
            elif input_data.entity_type == "Drug" or "药" in str(input_data.entity_name):
                mock_results = [
                    {
                        "entity_id": "drug_001",
                        "entity_label": "Drug",
                        "properties": {
                            "name": "阿司匹林",
                            "generic_name": "乙酰水杨酸",
                            "category": "analgesic",
                            "dosage_form": "片剂"
                        },
                        "score": 0.92
                    }
                ]
            elif input_data.entity_type == "Symptom" or "症状" in str(input_data.entity_name):
                mock_results = [
                    {
                        "entity_id": "symptom_001",
                        "entity_label": "Symptom",
                        "properties": {
                            "name": "头痛",
                            "severity": "moderate"
                        },
                        "score": 0.88
                    },
                    {
                        "entity_id": "symptom_002",
                        "entity_label": "Symptom",
                        "properties": {
                            "name": "发热",
                            "severity": "moderate"
                        },
                        "score": 0.85
                    }
                ]
        
        # 关系查询
        elif input_data.query_type == "relationship":
            mock_results = [
                {
                    "entity_id": "drug_003",
                    "entity_label": "Drug",
                    "properties": {
                        "name": "氨氯地平"
                    },
                    "score": 0.90,
                    "relationships": [
                        {
                            "type": "TREATS",
                            "source": "drug_003",
                            "target": "disease_001",
                            "properties": {"effectiveness": "high"}
                        }
                    ]
                }
            ]
        
        # 默认结果
        if not mock_results:
            mock_results = [
                {
                    "entity_id": "entity_unknown",
                    "entity_label": "Unknown",
                    "properties": {"name": "未找到相关实体"},
                    "score": 0.5
                }
            ]
        
        return KGQueryOutput(
            success=True,
            results=mock_results[:input_data.max_results],
            result_count=len(mock_results[:input_data.max_results]),
            query_info={
                "query_type": input_data.query_type,
                "entity_name": input_data.entity_name,
                "mock": True
            }
        )
    
    def query_disease_info(self, disease_name: str) -> KGQueryOutput:
        """
        便捷方法：查询疾病信息
        
        Args:
            disease_name: 疾病名称
            
        Returns:
            查询输出
        """
        input_data = KGQueryInput(
            query_type="entity",
            entity_type="Disease",
            entity_name=disease_name,
            max_results=5
        )
        return self.run(input_data)
    
    def query_drug_info(self, drug_name: str) -> KGQueryOutput:
        """
        便捷方法：查询药品信息
        
        Args:
            drug_name: 药品名称
            
        Returns:
            查询输出
        """
        input_data = KGQueryInput(
            query_type="entity",
            entity_type="Drug",
            entity_name=drug_name,
            max_results=5
        )
        return self.run(input_data)
    
    def query_disease_treatments(self, disease_id: str) -> KGQueryOutput:
        """
        便捷方法：查询疾病的治疗方法
        
        Args:
            disease_id: 疾病ID
            
        Returns:
            查询输出
        """
        input_data = KGQueryInput(
            query_type="relationship",
            source_entity=disease_id,
            relationship_type="TREATED_BY",
            max_results=10
        )
        return self.run(input_data)