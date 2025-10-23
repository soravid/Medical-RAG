"""
患者信息工具
提供患者档案查询和管理功能
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

logger = logging.getLogger(__name__)


class PatientInfoInput(BaseModel):
    """患者信息工具输入"""
    patient_id: str = Field(..., description="患者ID")
    info_type: str = Field(
        default="all",
        description="信息类型 (all/basic/profile/history/medications)"
    )
    include_history: bool = Field(default=False, description="是否包含病历历史")
    max_history_records: int = Field(default=5, description="最大病历记录数")


class PatientInfoOutput(BaseModel):
    """患者信息工具输出"""
    success: bool = Field(..., description="查询是否成功")
    patient_id: str = Field(..., description="患者ID")
    basic_info: Optional[Dict[str, Any]] = Field(None, description="基本信息")
    profile: Optional[Dict[str, Any]] = Field(None, description="健康档案")
    medical_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="病历历史"
    )
    current_medications: List[str] = Field(
        default_factory=list,
        description="当前用药"
    )
    allergies: List[str] = Field(default_factory=list, description="过敏史")
    warnings: List[str] = Field(default_factory=list, description="警告信息")
    error_message: Optional[str] = Field(None, description="错误信息")


class PatientInfoTool:
    """
    患者信息工具
    查询和管理患者档案信息
    """
    
    def __init__(
        self,
        patient_database: Optional[Any] = None,
        use_mock: bool = True
    ):
        """
        初始化患者信息工具
        
        Args:
            patient_database: 患者数据库实例
            use_mock: 是否使用模拟数据
        """
        self.patient_database = patient_database
        self.use_mock = use_mock
        
        if self.patient_database is None and not use_mock:
            logger.warning("未提供患者数据库，将使用模拟数据")
            self.use_mock = True
    
    @property
    def name(self) -> str:
        return "patient_info"
    
    @property
    def description(self) -> str:
        return (
            "查询患者档案信息，包括基本信息、健康档案、病历历史、当前用药、过敏史等。"
            "用于获取患者的详细信息以辅助诊断和治疗决策。"
        )
    
    def run(self, input_data: PatientInfoInput) -> PatientInfoOutput:
        """
        执行患者信息查询
        
        Args:
            input_data: 查询输入
            
        Returns:
            查询输出
        """
        logger.info(f"查询患者信息: patient_id={input_data.patient_id}, type={input_data.info_type}")
        
        try:
            if self.use_mock or self.patient_database is None:
                return self._mock_query(input_data)
            
            # 实际查询逻辑
            output = self._execute_query(input_data)
            return output
        
        except Exception as e:
            logger.error(f"患者信息查询失败: {e}")
            return PatientInfoOutput(
                success=False,
                patient_id=input_data.patient_id,
                error_message=str(e)
            )
    
    def _execute_query(self, input_data: PatientInfoInput) -> PatientInfoOutput:
        """执行实际查询"""
        # 实际实现需要连接患者数据库
        # 这里提供接口框架
        patient_id = input_data.patient_id
        
        basic_info = None
        profile = None
        medical_history = []
        current_medications = []
        allergies = []
        warnings = []
        
        if input_data.info_type in ["all", "basic"]:
            # 查询基本信息
            basic_info = self.patient_database.get_basic_info(patient_id)
        
        if input_data.info_type in ["all", "profile"]:
            # 查询健康档案
            profile = self.patient_database.get_profile(patient_id)
        
        if input_data.info_type in ["all", "history"] or input_data.include_history:
            # 查询病历历史
            medical_history = self.patient_database.get_medical_history(
                patient_id,
                limit=input_data.max_history_records
            )
        
        if input_data.info_type in ["all", "medications"]:
            # 查询当前用药
            current_medications = self.patient_database.get_current_medications(patient_id)
        
        # 提取过敏史
        if profile and "allergies" in profile:
            allergies = profile["allergies"]
        
        # 生成警告信息
        if allergies:
            warnings.append(f"患者有 {len(allergies)} 项过敏记录，用药时需特别注意")
        
        return PatientInfoOutput(
            success=True,
            patient_id=patient_id,
            basic_info=basic_info,
            profile=profile,
            medical_history=medical_history,
            current_medications=current_medications,
            allergies=allergies,
            warnings=warnings
        )
    
    def _mock_query(self, input_data: PatientInfoInput) -> PatientInfoOutput:
        """模拟查询"""
        logger.info("使用模拟患者信息")
        
        patient_id = input_data.patient_id
        
        # 模拟基本信息
        basic_info = {
            "patient_id": patient_id,
            "name": "张三",
            "age": 45,
            "gender": "male",
            "contact": "138****8888",
            "id_number": "320***********1234"
        } if input_data.info_type in ["all", "basic"] else None
        
        # 模拟健康档案
        profile = {
            "height": 175.0,
            "weight": 70.0,
            "bmi": 22.86,
            "blood_type": "A",
            "chronic_diseases": ["高血压"],
            "allergies": ["青霉素", "海鲜"],
            "family_history": {
                "父亲": ["糖尿病", "高血压"],
                "母亲": ["高血脂"]
            },
            "lifestyle": {
                "smoking": False,
                "drinking": "occasionally",
                "exercise": "moderate"
            }
        } if input_data.info_type in ["all", "profile"] else None
        
        # 模拟病历历史
        medical_history = []
        if input_data.info_type in ["all", "history"] or input_data.include_history:
            medical_history = [
                {
                    "visit_date": "2024-01-15",
                    "chief_complaint": "头晕、头痛",
                    "diagnosis": "高血压",
                    "prescribed_drugs": ["氨氯地平片"],
                    "notes": "血压控制良好，继续用药"
                },
                {
                    "visit_date": "2023-12-10",
                    "chief_complaint": "感冒症状",
                    "diagnosis": "急性上呼吸道感染",
                    "prescribed_drugs": ["对乙酰氨基酚片", "维生素C"],
                    "notes": "已康复"
                },
                {
                    "visit_date": "2023-10-20",
                    "chief_complaint": "体检",
                    "diagnosis": "健康体检",
                    "prescribed_drugs": [],
                    "notes": "发现血压偏高，建议监测"
                }
            ][:input_data.max_history_records]
        
        # 模拟当前用药
        current_medications = ["氨氯地平片 5mg 每日一次"] if input_data.info_type in ["all", "medications"] else []
        
        # 提取过敏史
        allergies = profile.get("allergies", []) if profile else []
        
        # 生成警告信息
        warnings = []
        if allergies:
            warnings.append(f"患者对以下物质过敏: {', '.join(allergies)}")
        if profile and "chronic_diseases" in profile and profile["chronic_diseases"]:
            warnings.append(f"患者有慢性疾病: {', '.join(profile['chronic_diseases'])}")
        
        return PatientInfoOutput(
            success=True,
            patient_id=patient_id,
            basic_info=basic_info,
            profile=profile,
            medical_history=medical_history,
            current_medications=current_medications,
            allergies=allergies,
            warnings=warnings
        )
    
    def get_allergies(self, patient_id: str) -> List[str]:
        """
        便捷方法：获取患者过敏史
        
        Args:
            patient_id: 患者ID
            
        Returns:
            过敏史列表
        """
        input_data = PatientInfoInput(
            patient_id=patient_id,
            info_type="profile"
        )
        output = self.run(input_data)
        return output.allergies
    
    def get_current_medications(self, patient_id: str) -> List[str]:
        """
        便捷方法：获取当前用药
        
        Args:
            patient_id: 患者ID
            
        Returns:
            当前用药列表
        """
        input_data = PatientInfoInput(
            patient_id=patient_id,
            info_type="medications"
        )
        output = self.run(input_data)
        return output.current_medications
    
    def get_medical_history(self, patient_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        便捷方法：获取病历历史
        
        Args:
            patient_id: 患者ID
            limit: 返回记录数
            
        Returns:
            病历历史列表
        """
        input_data = PatientInfoInput(
            patient_id=patient_id,
            info_type="history",
            max_history_records=limit
        )
        output = self.run(input_data)
        return output.medical_history