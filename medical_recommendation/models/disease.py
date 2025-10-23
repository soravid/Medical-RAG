"""
疾病数据模型定义
包含疾病基本信息、症状和诊断结果
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict


class DiseaseSeverity(str, Enum):
    """疾病严重程度枚举"""
    MILD = "mild"          # 轻度
    MODERATE = "moderate"  # 中度
    SEVERE = "severe"      # 重度
    CRITICAL = "critical"  # 危重


class DiseaseCategory(str, Enum):
    """疾病分类枚举"""
    INFECTIOUS = "infectious"              # 传染性疾病
    CARDIOVASCULAR = "cardiovascular"      # 心血管疾病
    RESPIRATORY = "respiratory"            # 呼吸系统疾病
    DIGESTIVE = "digestive"                # 消化系统疾病
    ENDOCRINE = "endocrine"                # 内分泌疾病
    NEUROLOGICAL = "neurological"          # 神经系统疾病
    MUSCULOSKELETAL = "musculoskeletal"    # 肌肉骨骼疾病
    METABOLIC = "metabolic"                # 代谢性疾病
    MENTAL = "mental"                      # 精神疾病
    CANCER = "cancer"                      # 肿瘤
    OTHER = "other"                        # 其他


class SymptomSeverity(str, Enum):
    """症状严重程度枚举"""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class Disease(BaseModel):
    """
    疾病基础信息模型
    
    Attributes:
        id: 疾病唯一标识符
        name: 疾病名称
        icd_code: ICD编码（国际疾病分类）
        category: 疾病分类
        description: 疾病描述
        common_symptoms: 常见症状列表
        risk_factors: 风险因素列表
        complications: 并发症列表
        typical_treatments: 典型治疗方法列表
        prognosis: 预后信息
        prevalence: 患病率信息
        created_at: 创建时间
    """
    model_config = ConfigDict(use_enum_values=True)
    
    id: str = Field(..., description="疾病唯一标识符", min_length=1)
    name: str = Field(..., description="疾病名称", min_length=1, max_length=200)
    icd_code: Optional[str] = Field(None, description="ICD编码")
    category: DiseaseCategory = Field(..., description="疾病分类")
    description: Optional[str] = Field(None, description="疾病描述")
    common_symptoms: List[str] = Field(
        default_factory=list,
        description="常见症状列表"
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="风险因素列表"
    )
    complications: List[str] = Field(
        default_factory=list,
         description="并发症列表"
    )
    typical_treatments: List[str] = Field(
        default_factory=list,
        description="典型治疗方法列表"
    )
    prognosis: Optional[str] = Field(None, description="预后信息")
    prevalence: Optional[str] = Field(None, description="患病率信息")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    
    @field_validator("common_symptoms", "risk_factors", "complications", "typical_treatments")
    @classmethod
    def validate_string_list(cls, v: List[str]) -> List[str]:
        """验证字符串列表，移除空字符串"""
        return [s.strip() for s in v if s and s.strip()]
    
    @field_validator("icd_code")
    @classmethod
    def validate_icd_code(cls, v: Optional[str]) -> Optional[str]:
        """验证ICD编码格式"""
        if v and v.strip():
            return v.strip().upper()
        return None
    
    def has_symptom(self, symptom: str) -> bool:
        """检查是否包含特定症状"""
        return any(
            symptom.lower() in s.lower() 
            for s in self.common_symptoms
        )
    
    def add_symptom(self, symptom: str) -> None:
        """添加症状"""
        if symptom and symptom.strip() and symptom not in self.common_symptoms:
            self.common_symptoms.append(symptom.strip())
    
    def __str__(self) -> str:
        return f"Disease(id={self.id}, name={self.name}, category={self.category})"
class Symptom(BaseModel):
    """
    症状信息模型
    
    Attributes:
        id: 症状唯一标识符
        name: 症状名称
        description: 症状描述
        severity: 症状严重程度
        body_part: 相关身体部位
        duration: 持续时间
        frequency: 发生频率
        onset: 发作方式（急性/慢性）
        aggravating_factors: 加重因素
        relieving_factors: 缓解因素
        associated_symptoms: 伴随症状
        recorded_at: 记录时间
    """
    model_config = ConfigDict(use_enum_values=True)
    
    id: str = Field(..., description="症状唯一标识符")
    name: str = Field(..., description="症状名称", min_length=1, max_length=100)
    description: Optional[str] = Field(None, description="症状描述")
    severity: SymptomSeverity = Field(
        default=SymptomSeverity.MILD,
        description="症状严重程度"
    )
    body_part: Optional[str] = Field(None, description="相关身体部位")
    duration: Optional[str] = Field(None, description="持续时间（如2天、1周等）")
    frequency: Optional[str] = Field(None, description="发生频率（如间歇性、持续性等）")
    onset: Optional[str] = Field(None, description="发作方式（急性/慢性/突然/逐渐）")
    aggravating_factors: List[str] = Field(
        default_factory=list,
        description="加重因素列表"
    )
    relieving_factors: List[str] = Field(
        default_factory=list,
        description="缓解因素列表"
    )
    associated_symptoms: List[str] = Field(
        default_factory=list,
        description="伴随症状列表"
    )
    recorded_at: datetime = Field(default_factory=datetime.now, description="记录时间")
    
    @field_validator("aggravating_factors", "relieving_factors", "associated_symptoms")
    @classmethod
    def validate_string_list(cls, v: List[str]) -> List[str]:
        """验证字符串列表"""
        return [s.strip() for s in v if s and s.strip()]
    
    def is_severe(self) -> bool:
        """判断是否为严重症状"""
        return self.severity == SymptomSeverity.SEVERE
    
    def add_associated_symptom(self, symptom: str) -> None:
        """添加伴随症状"""
        if symptom and symptom.strip() and symptom not in self.associated_symptoms:
            self.associated_symptoms.append(symptom.strip())
    
    def __str__(self) -> str:
        return f"Symptom(id={self.id}, name={self.name}, severity={self.severity})"
class DiagnosisResult(BaseModel):
    """
    诊断结果模型
    
    Attributes:
        id: 诊断记录唯一标识符
        patient_id: 患者ID
        disease_id: 疾病ID
        disease_name: 疾病名称
        confidence: 诊断置信度（0-1之间）
        severity: 疾病严重程度
        symptoms: 相关症状列表
        diagnosis_method: 诊断方法
        lab_results: 实验室检查结果
        imaging_results: 影像学检查结果
        differential_diagnoses: 鉴别诊断列表
        notes: 诊断备注
        diagnosed_by: 诊断医生
        diagnosed_at: 诊断时间
    """
    model_config = ConfigDict(use_enum_values=True)
    
    id: str = Field(..., description="诊断记录唯一标识符")
    patient_id: str = Field(..., description="患者ID")
    disease_id: str = Field(..., description="疾病ID")
    disease_name: str = Field(..., description="疾病名称", min_length=1)
    confidence: float = Field(
        ...,
        description="诊断置信度",
        ge=0.0,
        le=1.0
    )
    severity: DiseaseSeverity = Field(
        default=DiseaseSeverity.MILD,
        description="疾病严重程度"
    )
    symptoms: List[str] = Field(default_factory=list, description="相关症状列表")
    diagnosis_method: Optional[str] = Field(None, description="诊断方法")
    lab_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="实验室检查结果"
    )
    imaging_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="影像学检查结果"
    )
    differential_diagnoses: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="鉴别诊断列表，包含疾病名称和可能性"
    )
    notes: Optional[str] = Field(None, description="诊断备注")
    diagnosed_by: Optional[str] = Field(None, description="诊断医生ID或姓名")
    diagnosed_at: datetime = Field(default_factory=datetime.now, description="诊断时间")
    
    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """验证置信度范围"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("置信度必须在0到1之间")
        return round(v, 4)
    
    @field_validator("symptoms")
    @classmethod
    def validate_symptoms(cls, v: List[str]) -> List[str]:
        """验证症状列表"""
        return [s.strip() for s in v if s and s.strip()]
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """判断是否为高置信度诊断"""
        return self.confidence >= threshold
    
    def is_critical(self) -> bool:
        """判断是否为危重情况"""
        return self.severity == DiseaseSeverity.CRITICAL
    
    def add_differential_diagnosis(
        self,
        disease_name: str,
        probability: float,
        reasoning: Optional[str] = None
    ) -> None:
        """
        添加鉴别诊断
        
        Args:
            disease_name: 疾病名称
            probability: 可能性（0-1）
            reasoning: 诊断理由
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError("可能性必须在0到1之间")
        
        differential = {
            "disease_name": disease_name,
            "probability": round(probability, 4),
            "reasoning": reasoning
        }
        self.differential_diagnoses.append(differential)
    
    def add_lab_result(self, test_name: str, result: Any, unit: Optional[str] = None) -> None:
        """
        添加实验室检查结果
        
        Args:
            test_name: 检查项目名称
            result: 检查结果
            unit: 单位
        """
        self.lab_results[test_name] = {
            "result": result,
            "unit": unit,
            "recorded_at": datetime.now().isoformat()
        }
    
    def __str__(self) -> str:
        return (
            f"DiagnosisResult(id={self.id}, disease={self.disease_name}, "
            f"confidence={self.confidence:.2f}, severity={self.severity})"
        )