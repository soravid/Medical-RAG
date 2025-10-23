"""
患者数据模型定义
包含患者基本信息、医疗档案和病历信息
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict


class Gender(str, Enum):
    """性别枚举"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class BloodType(str, Enum):
    """血型枚举"""
    A = "A"
    B = "B"
    AB = "AB"
    O = "O"
    UNKNOWN = "unknown"


class AllergyLevel(str, Enum):
    """过敏等级枚举"""
    MILD = "mild"          # 轻度
    MODERATE = "moderate"  # 中度
    SEVERE = "severe"      # 重度


class Patient(BaseModel):
    """
    患者基础信息模型
    
    Attributes:
        id: 患者唯一标识符
        name: 患者姓名
        age: 患者年龄
        gender: 患者性别
        contact: 联系方式
        created_at: 创建时间
        updated_at: 更新时间
    """
    model_config = ConfigDict(use_enum_values=True)
    
    id: str = Field(..., description="患者唯一标识符", min_length=1)
    name: str = Field(..., description="患者姓名", min_length=1, max_length=100)
    age: int = Field(..., description="患者年龄", ge=0, le=150)
    gender: Gender = Field(..., description="患者性别")
    contact: Optional[str] = Field(None, description="联系方式（电话/邮箱）")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    
    @field_validator("contact")
    @classmethod
    def validate_contact(cls, v: Optional[str]) -> Optional[str]:
        """验证联系方式格式"""
        if v and len(v.strip()) == 0:
            return None
        return v
    
    def __str__(self) -> str:
        return f"Patient(id={self.id}, name={self.name}, age={self.age})"


class PatientProfile(BaseModel):
    """
    患者健康档案模型
    
    Attributes:
        patient_id: 关联的患者ID
        height: 身高（厘米）
        weight: 体重（公斤）
        blood_type: 血型
        allergies: 过敏史列表
        chronic_diseases: 慢性疾病列表
        family_history: 家族病史
        lifestyle: 生活习惯信息
        bmi: 体重指数（自动计算）
    """
    model_config = ConfigDict(use_enum_values=True)
    
    patient_id: str = Field(..., description="关联的患者ID")
    height: Optional[float] = Field(None, description="身高（厘米）", ge=0, le=300)
    weight: Optional[float] = Field(None, description="体重（公斤）", ge=0, le=500)
    blood_type: BloodType = Field(default=BloodType.UNKNOWN, description="血型")
    allergies: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="过敏史列表，包含过敏原和过敏等级"
    )
    chronic_diseases: List[str] = Field(
        default_factory=list,
        description="慢性疾病列表"
    )
    family_history: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="家族病史，key为关系，value为疾病列表"
    )
    lifestyle: Dict[str, Any] = Field(
        default_factory=dict,
        description="生活习惯信息（吸烟、饮酒、运动等）"
    )
    
    @property
    def bmi(self) -> Optional[float]:
        """计算体重指数（BMI）"""
        if self.height and self.weight and self.height > 0:
            height_m = self.height / 100
            return round(self.weight / (height_m ** 2), 2)
        return None
    
    def add_allergy(
        self, 
        allergen: str, 
        level: AllergyLevel = AllergyLevel.MILD,
        description: Optional[str] = None
    ) -> None:
        """
        添加过敏信息
        
        Args:
            allergen: 过敏原名称
            level: 过敏等级
            description: 过敏描述
        """
        allergy_info = {
            "allergen": allergen,
            "level": level.value if isinstance(level, AllergyLevel) else level,
            "description": description,
            "recorded_at": datetime.now().isoformat()
        }
        self.allergies.append(allergy_info)
    
    def has_allergy(self, allergen: str) -> bool:
        """检查是否对某种物质过敏"""
        return any(
            allergy.get("allergen", "").lower() == allergen.lower() 
            for allergy in self.allergies
        )


class MedicalHistory(BaseModel):
    """
    患者病历记录模型
    
    Attributes:
        id: 病历唯一标识符
        patient_id: 关联的患者ID
        visit_date: 就诊日期
        chief_complaint: 主诉
        symptoms: 症状列表
        diagnosis: 诊断结果
        prescribed_drugs: 开具的药品列表
        treatment_plan: 治疗方案
        notes: 医生备注
        follow_up_date: 复诊日期
    """
    model_config = ConfigDict(use_enum_values=True)
    
    id: str = Field(..., description="病历唯一标识符")
    patient_id: str = Field(..., description="关联的患者ID")
    visit_date: datetime = Field(default_factory=datetime.now, description="就诊日期")
    chief_complaint: str = Field(..., description="主诉", min_length=1)
    symptoms: List[str] = Field(default_factory=list, description="症状列表")
    diagnosis: Optional[str] = Field(None, description="诊断结果")
    prescribed_drugs: List[str] = Field(default_factory=list, description="开具的药品ID列表")
    treatment_plan: Optional[str] = Field(None, description="治疗方案")
    notes: Optional[str] = Field(None, description="医生备注")
    follow_up_date: Optional[datetime] = Field(None, description="复诊日期")
    
    @field_validator("symptoms")
    @classmethod
    def validate_symptoms(cls, v: List[str]) -> List[str]:
        """验证症状列表不为空字符串"""
        return [s.strip() for s in v if s and s.strip()]
    
    def add_symptom(self, symptom: str) -> None:
        """添加症状"""
        if symptom and symptom.strip() and symptom not in self.symptoms:
            self.symptoms.append(symptom.strip())
    
    def add_prescribed_drug(self, drug_id: str) -> None:
        """添加处方药品"""
        if drug_id and drug_id not in self.prescribed_drugs:
            self.prescribed_drugs.append(drug_id)