"""
患者画像管理
构建和管理患者的综合健康档案
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PatientProfile(BaseModel):
    """患者画像模型"""
    patient_id: str = Field(..., description="患者ID")
    basic_info: Dict[str, Any] = Field(default_factory=dict, description="基本信息")
    health_profile: Dict[str, Any] = Field(default_factory=dict, description="健康档案")
    risk_factors: List[str] = Field(default_factory=list, description="风险因素")
    chronic_conditions: List[str] = Field(default_factory=list, description="慢性病")
    allergies: List[str] = Field(default_factory=list, description="过敏史")
    current_medications: List[str] = Field(default_factory=list, description="当前用药")
    medical_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="病历历史"
    )
    lifestyle: Dict[str, Any] = Field(default_factory=dict, description="生活方式")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="偏好设置")
    last_updated: datetime = Field(default_factory=datetime.now, description="最后更新时间")
    
    def get_risk_score(self) -> float:
        """计算风险评分"""
        score = 0.0
        
        # 基于年龄
        age = self.basic_info.get("age", 0)
        if age > 60:
            score += 0.3
        elif age > 40:
            score += 0.2
        
        # 基于慢性病
        score += len(self.chronic_conditions) * 0.1
        
        # 基于风险因素
        score += len(self.risk_factors) * 0.05
        
        # 基于生活方式
        if self.lifestyle.get("smoking"):
            score += 0.2
        if self.lifestyle.get("drinking") == "heavy":
            score += 0.15
        
        return min(score, 1.0)
    
    def has_contraindication(self, drug: str) -> bool:
        """检查是否有药物禁忌"""
        # 检查过敏
        for allergy in self.allergies:
            if allergy.lower() in drug.lower():
                return True
        
        # 检查药物相互作用
        for current_med in self.current_medications:
            # 简化的相互作用检查
            if "相互作用" in current_med:
                return True
        
        return False


class PatientProfileManager:
    """
    患者画像管理器
    负责构建、更新和维护患者画像
    """
    
    def __init__(self, patient_database: Optional[Any] = None, use_mock: bool = True):
        """
        初始化患者画像管理器
        
        Args:
            patient_database: 患者数据库实例
            use_mock: 是否使用模拟数据
        """
        self.patient_database = patient_database
        self.use_mock = use_mock
        self._cache: Dict[str, PatientProfile] = {}
        
        if self.patient_database is None and not use_mock:
            logger.warning("未提供患者数据库，将使用模拟数据")
            self.use_mock = True
    
    def get_profile(self, patient_id: str, force_refresh: bool = False) -> Optional[PatientProfile]:
        """
        获取患者画像
        
        Args:
            patient_id: 患者ID
            force_refresh: 是否强制刷新
            
        Returns:
            患者画像
        """
        # 检查缓存
        if not force_refresh and patient_id in self._cache:
            logger.debug(f"从缓存获取患者画像: {patient_id}")
            return self._cache[patient_id]
        
        # 构建画像
        profile = self._build_profile(patient_id)
        
        if profile:
            self._cache[patient_id] = profile
            logger.info(f"患者画像已构建: {patient_id}")
        
        return profile
    
    def _build_profile(self, patient_id: str) -> Optional[PatientProfile]:
        """构建患者画像"""
        if self.use_mock or self.patient_database is None:
            return self._build_mock_profile(patient_id)
        
        try:
            # 从数据库获取患者信息
            basic_info = self.patient_database.get_basic_info(patient_id)
            health_profile = self.patient_database.get_health_profile(patient_id)
            medical_history = self.patient_database.get_medical_history(patient_id)
            
            # 分析风险因素
            risk_factors = self._analyze_risk_factors(basic_info, health_profile)
            
            # 构建画像
            profile = PatientProfile(
                patient_id=patient_id,
                basic_info=basic_info,
                health_profile=health_profile,
                risk_factors=risk_factors,
                chronic_conditions=health_profile.get("chronic_diseases", []),
                allergies=health_profile.get("allergies", []),
                current_medications=health_profile.get("current_medications", []),
                medical_history=medical_history,
                lifestyle=health_profile.get("lifestyle", {})
            )
            
            return profile
        
        except Exception as e:
            logger.error(f"构建患者画像失败: {e}")
            return None
    
    def _build_mock_profile(self, patient_id: str) -> PatientProfile:
        """构建模拟患者画像"""
        return PatientProfile(
            patient_id=patient_id,
            basic_info={
                "name": "张三",
                "age": 45,
                "gender": "male",
                "height": 175,
                "weight": 70
            },
            health_profile={
                "blood_type": "A",
                "bmi": 22.86
            },
            risk_factors=["高血压家族史", "工作压力大"],
            chronic_conditions=["高血压"],
            allergies=["青霉素"],
            current_medications=["氨氯地平片 5mg 每日一次"],
            lifestyle={
                "smoking": False,
                "drinking": "occasionally",
                "exercise": "moderate"
            }
        )
    
    def _analyze_risk_factors(
        self,
        basic_info: Dict[str, Any],
        health_profile: Dict[str, Any]
    ) -> List[str]:
        """分析风险因素"""
        risk_factors = []
        
        # 年龄风险
        age = basic_info.get("age", 0)
        if age > 60:
            risk_factors.append("高龄")
        
        # BMI风险
        bmi = health_profile.get("bmi", 0)
        if bmi > 28:
            risk_factors.append("肥胖")
        elif bmi > 24:
            risk_factors.append("超重")
        
        # 生活方式风险
        lifestyle = health_profile.get("lifestyle", {})
        if lifestyle.get("smoking"):
            risk_factors.append("吸烟")
        if lifestyle.get("drinking") in ["heavy", "frequent"]:
            risk_factors.append("过量饮酒")
        
        # 家族史风险
        family_history = health_profile.get("family_history", {})
        if family_history:
            risk_factors.append("家族病史")
        
        return risk_factors
    
    def update_profile(
        self,
        patient_id: str,
        updates: Dict[str, Any]
    ) -> Optional[PatientProfile]:
        """
        更新患者画像
        
        Args:
            patient_id: 患者ID
            updates: 更新数据
            
        Returns:
            更新后的画像
        """
        profile = self.get_profile(patient_id)
        
        if not profile:
            logger.warning(f"患者画像不存在: {patient_id}")
            return None
        
        # 更新字段
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        profile.last_updated = datetime.now()
        
        # 更新缓存
        self._cache[patient_id] = profile
        
        logger.info(f"患者画像已更新: {patient_id}")
        return profile
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        logger.info("患者画像缓存已清空")