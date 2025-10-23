"""
药品数据模型定义
包含药品基本信息、药物相互作用和处方信息
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict


class DrugCategory(str, Enum):
    """药品分类枚举"""
    ANTIBIOTIC = "antibiotic"              # 抗生素
    ANALGESIC = "analgesic"                # 镇痛药
    ANTIPYRETIC = "antipyretic"            # 退烧药
    ANTIVIRAL = "antiviral"                # 抗病毒药
    CARDIOVASCULAR = "cardiovascular"      # 心血管药
    GASTROINTESTINAL = "gastrointestinal"  # 胃肠道药
    RESPIRATORY = "respiratory"            # 呼吸系统药
    ENDOCRINE = "endocrine"                # 内分泌药
    VITAMIN = "vitamin"                    # 维生素
    OTHER = "other"                        # 其他


class PrescriptionType(str, Enum):
    """处方类型枚举"""
    OTC = "otc"          # 非处方药（Over The Counter）
    RX = "rx"            # 处方药
    CONTROLLED = "controlled"  # 管制药品


class InteractionSeverity(str, Enum):
    """药物相互作用严重程度枚举"""
    MINOR = "minor"          # 轻微
    MODERATE = "moderate"    # 中等
    MAJOR = "major"          # 严重
    CONTRAINDICATED = "contraindicated"  # 禁忌


class Drug(BaseModel):
    """
    药品基础信息模型
    
    Attributes:
        id: 药品唯一标识符
        name: 药品名称
        generic_name: 通用名称
        category: 药品分类
        prescription_type: 处方类型
        description: 药品描述
        indications: 适应症列表
        contraindications: 禁忌症列表
        side_effects: 副作用列表
        dosage_form: 剂型（片剂、胶囊等）
        strength: 规格/浓度
        manufacturer: 生产厂家
        created_at: 创建时间
    """
    model_config = ConfigDict(use_enum_values=True)
    
    id: str = Field(..., description="药品唯一标识符", min_length=1)
    name: str = Field(..., description="药品名称", min_length=1, max_length=200)
    generic_name: Optional[str] = Field(None, description="通用名称")
    category: DrugCategory = Field(..., description="药品分类")
    prescription_type: PrescriptionType = Field(
        default=PrescriptionType.RX,
        description="处方类型"
    )
    description: Optional[str] = Field(None, description="药品描述")
    indications: List[str] = Field(default_factory=list, description="适应症列表")
    contraindications: List[str] = Field(default_factory=list, description="禁忌症列表")
    side_effects: List[str] = Field(default_factory=list, description="副作用列表")
    dosage_form: Optional[str] = Field(None, description="剂型（片剂、胶囊、注射液等）")
    strength: Optional[str] = Field(None, description="规格/浓度（如100mg、5ml等）")
    manufacturer: Optional[str] = Field(None, description="生产厂家")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    
    @field_validator("indications", "contraindications", "side_effects")
    @classmethod
    def validate_string_list(cls, v: List[str]) -> List[str]:
        """验证字符串列表，移除空字符串"""
        return [s.strip() for s in v if s and s.strip()]
    
    def is_otc(self) -> bool:
        """判断是否为非处方药"""
        return self.prescription_type == PrescriptionType.OTC
    
    def has_contraindication(self, condition: str) -> bool:
        """检查是否有特定禁忌症"""
        return any(
            condition.lower() in ci.lower() 
            for ci in self.contraindications
        )
    
    def __str__(self) -> str:
        return f"Drug(id={self.id}, name={self.name}, category={self.category})"


class DrugInteraction(BaseModel):
    """
    药物相互作用模型
    
    Attributes:
        id: 相互作用记录ID
        drug_id_1: 第一个药品ID
        drug_id_2: 第二个药品ID
        severity: 相互作用严重程度
        description: 相互作用描述
        mechanism: 作用机制
        clinical_effects: 临床表现
        management: 管理建议
    """
    model_config = ConfigDict(use_enum_values=True)
    
    id: str = Field(..., description="相互作用记录ID")
    drug_id_1: str = Field(..., description="第一个药品ID")
    drug_id_2: str = Field(..., description="第二个药品ID")
    severity: InteractionSeverity = Field(..., description="相互作用严重程度")
    description: str = Field(..., description="相互作用描述", min_length=1)
    mechanism: Optional[str] = Field(None, description="作用机制")
    clinical_effects: List[str] = Field(
        default_factory=list,
        description="临床表现列表"
    )
    management: Optional[str] = Field(None, description="管理建议")
    
    @field_validator("drug_id_1", "drug_id_2")
    @classmethod
    def validate_drug_ids(cls, v: str) -> str:
        """验证药品ID不为空"""
        if not v or not v.strip():
            raise ValueError("药品ID不能为空")
        return v.strip()
    
    def is_severe(self) -> bool:
        """判断是否为严重相互作用"""
        return self.severity in [
            InteractionSeverity.MAJOR,
            InteractionSeverity.CONTRAINDICATED
        ]
    
    def involves_drug(self, drug_id: str) -> bool:
        """判断是否涉及指定药品"""
        return drug_id in [self.drug_id_1, self.drug_id_2]


class Prescription(BaseModel):
    """
    处方信息模型
    
    Attributes:
        id: 处方唯一标识符
        patient_id: 患者ID
        drug_id: 药品ID
        dosage: 剂量
        frequency: 用药频率
        duration: 用药时长
        route: 给药途径
        instructions: 用药说明
        prescribed_by: 开方医生
        prescribed_date: 开方日期
        start_date: 开始用药日期
        end_date: 结束用药日期
        refills: 可续方次数
        notes: 备注
    """
    model_config = ConfigDict(use_enum_values=True)
    
    id: str = Field(..., description="处方唯一标识符")
    patient_id: str = Field(..., description="患者ID")
    drug_id: str = Field(..., description="药品ID")
    dosage: str = Field(..., description="剂量（如1片、10ml等）")
    frequency: str = Field(..., description="用药频率（如每日3次、每8小时等）")
    duration: Optional[str] = Field(None, description="用药时长（如7天、2周等）")
    route: str = Field(default="oral", description="给药途径（口服、注射等）")
    instructions: Optional[str] = Field(None, description="用药说明")
    prescribed_by: str = Field(..., description="开方医生ID或姓名")
    prescribed_date: datetime = Field(default_factory=datetime.now, description="开方日期")
    start_date: Optional[datetime] = Field(None, description="开始用药日期")
    end_date: Optional[datetime] = Field(None, description="结束用药日期")
    refills: int = Field(default=0, description="可续方次数", ge=0)
    notes: Optional[str] = Field(None, description="备注")
    
    @field_validator("dosage", "frequency", "route")
    @classmethod
    def validate_required_fields(cls, v: str) -> str:
        """验证必填字段不为空"""
        if not v or not v.strip():
            raise ValueError("该字段不能为空")
        return v.strip()
    
    def is_active(self) -> bool:
        """判断处方是否有效"""
        now = datetime.now()
        if self.start_date and self.start_date > now:
            return False
        if self.end_date and self.end_date < now:
            return False
        return True
    
    def __str__(self) -> str:
        return f"Prescription(id={self.id}, patient={self.patient_id}, drug={self.drug_id})"