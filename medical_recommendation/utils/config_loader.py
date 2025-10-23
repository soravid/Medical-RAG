"""
配置加载器模块
提供YAML配置文件读取、验证和管理功能
"""

import os
import yaml
from typing import Any, Dict, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ValidationError
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeGraphConfig(BaseModel):
    """知识图谱配置模型"""
    host: str = Field(default="localhost", description="Neo4j主机地址")
    port: int = Field(default=7687, description="Neo4j端口")
    username: str = Field(default="neo4j", description="用户名")
    password: str = Field(..., description="密码")
    database: str = Field(default="neo4j", description="数据库名称")
    max_connection_lifetime: int = Field(default=3600, description="最大连接生命周期（秒）")
    max_connection_pool_size: int = Field(default=50, description="最大连接池大小")
    connection_timeout: int = Field(default=30, description="连接超时时间（秒）")
    
    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """验证端口号范围"""
        if not 1 <= v <= 65535:
            raise ValueError("端口号必须在1-65535之间")
        return v


class VectorStoreConfig(BaseModel):
    """向量存储配置模型"""
    type: str = Field(default="chromadb", description="向量数据库类型")
    host: Optional[str] = Field(default="localhost", description="主机地址")
    port: Optional[int] = Field(default=8000, description="端口")
    collection_name: str = Field(default="medical_documents", description="集合名称")
    persist_directory: Optional[str] = Field(default="./data/vector_db", description="持久化目录")
    distance_metric: str = Field(default="cosine", description="距离度量方式")
    embedding_dimension: int = Field(default=768, description="嵌入向量维度")
    
    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """验证向量数据库类型"""
        supported_types = ["chromadb", "faiss", "pinecone", "milvus"]
        if v.lower() not in supported_types:
            raise ValueError(f"不支持的向量数据库类型: {v}。支持的类型: {supported_types}")
        return v.lower()
    
    @field_validator("distance_metric")
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        """验证距离度量方式"""
        supported_metrics = ["cosine", "euclidean", "dot_product"]
        if v.lower() not in supported_metrics:
            raise ValueError(f"不支持的距离度量方式: {v}。支持的方式: {supported_metrics}")
        return v.lower()


class LLMConfig(BaseModel):
    """大语言模型配置模型"""
    provider: str = Field(default="openai", description="LLM提供商")
    model_name: str = Field(default="gpt-3.5-turbo", description="模型名称")
    api_key: Optional[str] = Field(None, description="API密钥")
    api_base: Optional[str] = Field(None, description="API基础URL")
    temperature: float = Field(default=0.7, description="温度参数")
    max_tokens: int = Field(default=2000, description="最大生成token数")
    top_p: float = Field(default=1.0, description="Top-p采样参数")
    frequency_penalty: float = Field(default=0.0, description="频率惩罚")
    presence_penalty: float = Field(default=0.0, description="存在惩罚")
    timeout: int = Field(default=60, description="请求超时时间（秒）")
    max_retries: int = Field(default=3, description="最大重试次数")
    
    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """验证温度参数"""
        if not 0.0 <= v <= 2.0:
            raise ValueError("温度参数必须在0.0-2.0之间")
        return v
    
    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, v: float) -> float:
        """验证top_p参数"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("top_p参数必须在0.0-1.0之间")
        return v
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """验证LLM提供商"""
        supported_providers = ["openai", "azure", "anthropic", "huggingface", "local"]
        if v.lower() not in supported_providers:
            raise ValueError(f"不支持的LLM提供商: {v}。支持的提供商: {supported_providers}")
        return v.lower()


class EmbeddingConfig(BaseModel):
    """嵌入模型配置模型"""
    provider: str = Field(default="openai", description="嵌入模型提供商")
    model_name: str = Field(default="text-embedding-ada-002", description="模型名称")
    api_key: Optional[str] = Field(None, description="API密钥")
    dimension: int = Field(default=1536, description="嵌入向量维度")
    batch_size: int = Field(default=100, description="批处理大小")
    max_retries: int = Field(default=3, description="最大重试次数")
    
    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, v: int) -> int:
        """验证嵌入向量维度"""
        if v <= 0:
            raise ValueError("嵌入向量维度必须大于0")
        return v


class ChainConfig(BaseModel):
    """链配置模型"""
    retrieval_k: int = Field(default=5, description="检索返回的文档数量")
    score_threshold: float = Field(default=0.7, description="相似度分数阈值")
    max_context_length: int = Field(default=4000, description="最大上下文长度")
    enable_memory: bool = Field(default=True, description="是否启用记忆")
    memory_window: int = Field(default=5, description="记忆窗口大小")
    verbose: bool = Field(default=False, description="是否输出详细日志")
    
    @field_validator("retrieval_k")
    @classmethod
    def validate_retrieval_k(cls, v: int) -> int:
        """验证检索数量"""
        if v <= 0:
            raise ValueError("检索数量必须大于0")
        return v
    
    @field_validator("score_threshold")
    @classmethod
    def validate_score_threshold(cls, v: float) -> float:
        """验证分数阈值"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("分数阈值必须在0.0-1.0之间")
        return v


class SystemConfig(BaseModel):
    """系统配置模型"""
    project_name: str = Field(default="MedicalRecommendationSystem", description="项目名称")
    version: str = Field(default="0.1.0", description="版本号")
    environment: str = Field(default="development", description="运行环境")
    debug: bool = Field(default=True, description="调试模式")
    log_level: str = Field(default="INFO", description="日志级别")
    data_dir: str = Field(default="./data", description="数据目录")
    cache_dir: str = Field(default="./cache", description="缓存目录")
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """验证运行环境"""
        valid_envs = ["development", "testing", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"无效的运行环境: {v}。有效值: {valid_envs}")
        return v.lower()
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """验证日志级别"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"无效的日志级别: {v}。有效值: {valid_levels}")
        return v_upper


class ConfigLoader:
    """
    配置加载器
    负责从YAML文件加载配置并进行验证
    """
    
    def __init__(self, config_dir: str = "./config"):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录路径
        """
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Any] = {}
        self._validated_configs: Dict[str, BaseModel] = {}
        
    def load_yaml(self, file_path: str) -> Dict[str, Any]:
        """
        从YAML文件加载配置
        
        Args:
            file_path: YAML文件路径
            
        Returns:
            配置字典
            
        Raises:
            FileNotFoundError: 文件不存在
            yaml.YAMLError: YAML解析错误
        """
        full_path = self.config_dir / file_path if not Path(file_path).is_absolute() else Path(file_path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {full_path}")
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"成功加载配置文件: {full_path}")
                return config or {}
        except yaml.YAMLError as e:
            logger.error(f"YAML解析错误: {e}")
            raise
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        加载所有配置文件
        
        Returns:
            所有配置的字典
        """
        config_files = {
            "kg": "kg_config.yaml",
            "vector": "vector_config.yaml",
            "llm": "llm_config.yaml",
            "chain": "chain_config.yaml",
            "system": "system_config.yaml"
        }
        
        all_configs = {}
        for key, filename in config_files.items():
            try:
                config = self.load_yaml(filename)
                all_configs[key] = config
                self._configs[key] = config
            except FileNotFoundError:
                logger.warning(f"配置文件未找到: {filename}，使用默认配置")
                all_configs[key] = {}
        
        return all_configs
    
    def validate_config(self, config_type: str, config_data: Dict[str, Any]) -> BaseModel:
        """
        验证配置数据
        
        Args:
            config_type: 配置类型 (kg, vector, llm, embedding, chain, system)
            config_data: 配置数据字典
            
        Returns:
            验证后的配置模型实例
            
        Raises:
            ValueError: 不支持的配置类型
            ValidationError: 配置验证失败
        """
        config_models = {
            "kg": KnowledgeGraphConfig,
            "vector": VectorStoreConfig,
            "llm": LLMConfig,
            "embedding": EmbeddingConfig,
            "chain": ChainConfig,
            "system": SystemConfig
        }
        
        if config_type not in config_models:
            raise ValueError(f"不支持的配置类型: {config_type}")
        
        model_class = config_models[config_type]
        
        try:
            validated_config = model_class(**config_data)
            self._validated_configs[config_type] = validated_config
            logger.info(f"配置验证成功: {config_type}")
            return validated_config
        except ValidationError as e:
            logger.error(f"配置验证失败 ({config_type}): {e}")
            raise
    
    def get_config(self, config_type: str, validated: bool = True) -> Any:
        """
        获取指定类型的配置
        
        Args:
            config_type: 配置类型
            validated: 是否返回验证后的配置模型
            
        Returns:
            配置数据或配置模型
        """
        if validated and config_type in self._validated_configs:
            return self._validated_configs[config_type]
        
        return self._configs.get(config_type, {})
    
    def save_config(self, config_type: str, config_data: Dict[str, Any], file_path: Optional[str] = None) -> None:
        """
        保存配置到YAML文件
        
        Args:
            config_type: 配置类型
            config_data: 配置数据
            file_path: 保存路径（可选）
        """
        if file_path is None:
            file_path = f"{config_type}_config.yaml"
        
        full_path = self.config_dir / file_path if not Path(file_path).is_absolute() else Path(file_path)
        
        # 确保目录存在
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_data, f, allow_unicode=True, default_flow_style=False)
            logger.info(f"配置已保存到: {full_path}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            raise
    
    def update_config(self, config_type: str, updates: Dict[str, Any]) -> None:
        """
        更新配置
        
        Args:
            config_type: 配置类型
            updates: 要更新的配置项
        """
        if config_type not in self._configs:
            self._configs[config_type] = {}
        
        self._configs[config_type].update(updates)
        
        # 重新验证
        try:
            self.validate_config(config_type, self._configs[config_type])
        except ValidationError:
            logger.warning(f"更新后的配置验证失败: {config_type}")
    
    def get_env_variable(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        获取环境变量
        
        Args:
            key: 环境变量名
            default: 默认值
            
        Returns:
            环境变量值
        """
        return os.getenv(key, default)
    
    def load_secrets_from_env(self, config_type: str, secret_keys: List[str]) -> None:
        """
        从环境变量加载敏感信息
        
        Args:
            config_type: 配置类型
            secret_keys: 敏感信息的键列表
        """
        if config_type not in self._configs:
            self._configs[config_type] = {}
        
        for key in secret_keys:
            env_key = f"{config_type.upper()}_{key.upper()}"
            value = self.get_env_variable(env_key)
            if value:
                self._configs[config_type][key] = value
                logger.info(f"从环境变量加载: {env_key}")


# 全局配置加载器实例
_global_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_dir: str = "./config") -> ConfigLoader:
    """
    获取全局配置加载器实例（单例模式）
    
    Args:
        config_dir: 配置文件目录
        
    Returns:
        ConfigLoader实例
    """
    global _global_config_loader
    if _global_config_loader is None:
        _global_config_loader = ConfigLoader(config_dir)
    return _global_config_loader


def get_config(config_type: str, validated: bool = True) -> Any:
    """
    便捷函数：获取指定类型的配置
    
    Args:
        config_type: 配置类型
        validated: 是否返回验证后的配置
        
    Returns:
        配置数据
    """
    loader = get_config_loader()
    return loader.get_config(config_type, validated)