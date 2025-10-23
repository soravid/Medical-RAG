"""
数据处理器模块
提供数据预处理、清洗、转换等功能
"""

import re
import json
from typing import List, Dict, Any, Optional, Union, Callable
from abc import ABC, abstractmethod
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataProcessor(ABC):
    """数据处理器抽象基类"""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        处理数据
        
        Args:
            data: 输入数据
            
        Returns:
            处理后的数据
        """
        pass
    
    def batch_process(self, data_list: List[Any]) -> List[Any]:
        """
        批量处理数据
        
        Args:
            data_list: 输入数据列表
            
        Returns:
            处理后的数据列表
        """
        return [self.process(data) for data in data_list]


class TextProcessor(DataProcessor):
    """
    文本处理器
    提供文本清洗、分词、标准化等功能
    """
    
    def __init__(
        self, 
        lowercase: bool = True,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
        remove_extra_whitespace: bool = True,
        custom_stopwords: Optional[List[str]] = None
    ):
        """
        初始化文本处理器
        
        Args:
            lowercase: 是否转换为小写
            remove_punctuation: 是否移除标点符号
            remove_numbers: 是否移除数字
            remove_extra_whitespace: 是否移除多余空白字符
            custom_stopwords: 自定义停用词列表
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_extra_whitespace = remove_extra_whitespace
        self.custom_stopwords = set(custom_stopwords) if custom_stopwords else set()
    
    def process(self, text: str) -> str:
        """
        处理文本
        
        Args:
            text: 输入文本
            
        Returns:
            处理后的文本
        """
        if not isinstance(text, str):
            logger.warning(f"输入不是字符串类型: {type(text)}")
            return str(text)
        
        result = text
        
        # 移除多余空白字符
        if self.remove_extra_whitespace:
            result = self.clean_whitespace(result)
        
        # 转换为小写
        if self.lowercase:
            result = result.lower()
        
        # 移除数字
        if self.remove_numbers:
            result = re.sub(r'\d+', '', result)
        
        # 移除标点符号
        if self.remove_punctuation:
            result = re.sub(r'[^\w\s]', '', result)
        
        # 移除停用词
        if self.custom_stopwords:
            words = result.split()
            words = [w for w in words if w not in self.custom_stopwords]
            result = ' '.join(words)
        
        return result.strip()
    
    @staticmethod
    def clean_whitespace(text: str) -> str:
        """
        清理多余的空白字符
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
        """
        # 替换多个空格为单个空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """
        移除HTML标签
        
        Args:
            text: 输入文本
            
        Returns:
            移除标签后的文本
        """
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """
        移除URL链接
        
        Args:
            text: 输入文本
            
        Returns:
            移除URL后的文本
        """
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return url_pattern.sub('', text)
    
    @staticmethod
    def remove_emails(text: str) -> str:
        """
        移除邮箱地址
        
        Args:
            text: 输入文本
            
        Returns:
            移除邮箱后的文本
        """
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        return email_pattern.sub('', text)
    
    def truncate(self, text: str, max_length: int, suffix: str = "...") -> str:
        """
        截断文本到指定长度
        
        Args:
            text: 输入文本
            max_length: 最大长度
            suffix: 截断后缀
            
        Returns:
            截断后的文本
        """
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        将文本分割为句子
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        # 简单的句子分割（支持中英文）
        sentence_endings = r'[。！？.!?]+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]


class MedicalDataProcessor(DataProcessor):
    """
    医疗数据处理器
    专门处理医疗领域的数据
    """
    
    def __init__(self):
        """初始化医疗数据处理器"""
        self.text_processor = TextProcessor(lowercase=False)
        
        # 医疗术语缩写映射
        self.abbreviation_map = {
            "BP": "blood pressure",
            "HR": "heart rate",
            "temp": "temperature",
            "dx": "diagnosis",
            "tx": "treatment",
            "rx": "prescription",
            "hx": "history",
        }
        
        # 医疗单位标准化
        self.unit_map = {
            "mg/dl": "mg/dL",
            "mmol/l": "mmol/L",
            "ml": "mL",
            "mcg": "μg",
        }
    
    def process(self, data: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """
        处理医疗数据
        
        Args:
            data: 输入数据（文本或字典）
            
        Returns:
            处理后的数据
        """
        if isinstance(data, str):
            return self.process_medical_text(data)
        elif isinstance(data, dict):
            return self.process_medical_record(data)
        else:
            logger.warning(f"不支持的数据类型: {type(data)}")
            return data
    
    def process_medical_text(self, text: str) -> str:
        """
        处理医疗文本
        
        Args:
            text: 输入医疗文本
            
        Returns:
            处理后的文本
        """
        # 基础文本清理
        text = self.text_processor.clean_whitespace(text)
        
        # 扩展缩写
        text = self.expand_abbreviations(text)
        
        # 标准化单位
        text = self.standardize_units(text)
        
        return text
    
    def process_medical_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理医疗记录
        
        Args:
            record: 医疗记录字典
            
        Returns:
            处理后的记录
        """
        processed = {}
        
        for key, value in record.items():
            if isinstance(value, str):
                processed[key] = self.process_medical_text(value)
            elif isinstance(value, list):
                processed[key] = [
                    self.process_medical_text(v) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                processed[key] = value
        
        return processed
    
    def expand_abbreviations(self, text: str) -> str:
        """
        扩展医疗缩写
        
        Args:
            text: 输入文本
            
        Returns:
            扩展后的文本
        """
        result = text
        for abbr, full in self.abbreviation_map.items():
            # 使用单词边界确保完整匹配
            pattern = r'\b' + re.escape(abbr) + r'\b'
            result = re.sub(pattern, full, result, flags=re.IGNORECASE)
        
        return result
    
    def standardize_units(self, text: str) -> str:
        """
        标准化医疗单位
        
        Args:
            text: 输入文本
            
        Returns:
            标准化后的文本
        """
        result = text
        for old_unit, new_unit in self.unit_map.items():
            result = result.replace(old_unit, new_unit)
        
        return result
    
    def extract_vital_signs(self, text: str) -> Dict[str, Optional[str]]:
        """
        从文本中提取生命体征
        
        Args:
            text: 输入医疗文本
            
        Returns:
            生命体征字典
        """
        vital_signs = {
            "blood_pressure": None,
            "heart_rate": None,
            "temperature": None,
            "respiratory_rate": None,
        }
        
        # 血压模式: 120/80 mmHg
        bp_pattern = r'(\d{2,3})/(\d{2,3})\s*mmHg'
        bp_match = re.search(bp_pattern, text)
        if bp_match:
            vital_signs["blood_pressure"] = f"{bp_match.group(1)}/{bp_match.group(2)} mmHg"
        
        # 心率模式: 72 bpm
        hr_pattern = r'(\d{2,3})\s*bpm'
        hr_match = re.search(hr_pattern, text, re.IGNORECASE)
        if hr_match:
            vital_signs["heart_rate"] = f"{hr_match.group(1)} bpm"
        
        # 体温模式: 36.5°C or 98.6°F
        temp_pattern = r'(\d{2,3}\.?\d*)\s*°?[CF]'
        temp_match = re.search(temp_pattern, text)
        if temp_match:
            vital_signs["temperature"] = temp_match.group(0)
        
        return vital_signs
    
    def anonymize_patient_info(
        self, 
        text: str, 
        replacement: str = "[REDACTED]"
    ) -> str:
        """
                匿名化患者敏感信息
        
        Args:
            text: 输入文本
            replacement: 替换文本
            
        Returns:
            匿名化后的文本
        """
        result = text
        
        # 移除电话号码
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\b1[3-9]\d{9}\b'
        result = re.sub(phone_pattern, replacement, result)
        
        # 移除邮箱
        result = self.text_processor.remove_emails(result)
        result = result.replace('', replacement)
        
        # 移除身份证号（中国）
        id_pattern = r'\b\d{17}[\dXx]\b'
        result = re.sub(id_pattern, replacement, result)
        
        return result
    
    def normalize_drug_name(self, drug_name: str) -> str:
        """
        标准化药品名称
        
        Args:
            drug_name: 药品名称
            
        Returns:
            标准化后的名称
        """
        # 移除剂型后缀
        suffixes = ['片', '胶囊', '颗粒', '注射液', '口服液', '糖浆']
        normalized = drug_name.strip()
        
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
        
        return normalized.strip()
    
    def parse_dosage(self, dosage_text: str) -> Dict[str, Any]:
        """
        解析用药剂量信息
        
        Args:
            dosage_text: 剂量文本，如 "100mg, 每日3次"
            
        Returns:
            解析后的剂量信息字典
        """
        dosage_info = {
            "amount": None,
            "unit": None,
            "frequency": None,
            "original": dosage_text
        }
        
        # 提取剂量和单位
        amount_pattern = r'(\d+\.?\d*)\s*(mg|g|ml|μg|mcg|片|粒)'
        amount_match = re.search(amount_pattern, dosage_text, re.IGNORECASE)
        if amount_match:
            dosage_info["amount"] = amount_match.group(1)
            dosage_info["unit"] = amount_match.group(2)
        
        # 提取频率
        frequency_patterns = [
            r'每日(\d+)次',
            r'每天(\d+)次',
            r'一天(\d+)次',
            r'(\d+)次/日',
        ]
        
        for pattern in frequency_patterns:
            freq_match = re.search(pattern, dosage_text)
            if freq_match:
                dosage_info["frequency"] = f"{freq_match.group(1)}次/日"
                break
        
        return dosage_info


class JSONDataProcessor(DataProcessor):
    """JSON数据处理器"""
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        初始化JSON数据处理器
        
        Args:
            encoding: 文件编码
        """
        self.encoding = encoding
    
    def process(self, data: Union[str, Dict, List]) -> Any:
        """
        处理JSON数据
        
        Args:
            data: 输入数据
            
        Returns:
            处理后的数据
        """
        if isinstance(data, str):
            return self.load_from_string(data)
        return data
    
    def load_from_file(self, file_path: str) -> Any:
        """
        从文件加载JSON数据
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            解析后的数据
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        try:
            with open(path, 'r', encoding=self.encoding) as f:
                data = json.load(f)
            logger.info(f"成功加载JSON文件: {file_path}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            raise
        except Exception as e:
            logger.error(f"加载文件失败: {e}")
            raise
    
    def load_from_string(self, json_string: str) -> Any:
        """
        从字符串加载JSON数据
        
        Args:
            json_string: JSON字符串
            
        Returns:
            解析后的数据
        """
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            raise
    
    def save_to_file(self, data: Any, file_path: str, indent: int = 2) -> None:
        """
        保存数据到JSON文件
        
        Args:
            data: 要保存的数据
            file_path: 保存路径
            indent: 缩进空格数
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w', encoding=self.encoding) as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
            logger.info(f"数据已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存文件失败: {e}")
            raise
    
    def validate_schema(self, data: Dict, required_fields: List[str]) -> bool:
        """
        验证JSON数据结构
        
        Args:
            data: 要验证的数据
            required_fields: 必需字段列表
            
        Returns:
            是否通过验证
        """
        if not isinstance(data, dict):
            logger.error("数据不是字典类型")
            return False
        
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            logger.error(f"缺少必需字段: {missing_fields}")
            return False
        
        return True


class DataPipeline:
    """
    数据处理流水线
    支持链式调用多个处理器
    """
    
    def __init__(self, processors: Optional[List[DataProcessor]] = None):
        """
        初始化数据流水线
        
        Args:
            processors: 处理器列表
        """
        self.processors: List[DataProcessor] = processors or []
    
    def add_processor(self, processor: DataProcessor) -> 'DataPipeline':
        """
        添加处理器
        
        Args:
            processor: 数据处理器
            
        Returns:
            self，支持链式调用
        """
        self.processors.append(processor)
        return self
    
    def add_function(self, func: Callable[[Any], Any]) -> 'DataPipeline':
        """
        添加自定义处理函数
        
        Args:
            func: 处理函数
            
        Returns:
            self，支持链式调用
        """
        class FunctionProcessor(DataProcessor):
            def __init__(self, f):
                self.func = f
            
            def process(self, data):
                return self.func(data)
        
        self.processors.append(FunctionProcessor(func))
        return self
    
    def process(self, data: Any) -> Any:
        """
        通过流水线处理数据
        
        Args:
            data: 输入数据
            
        Returns:
            处理后的数据
        """
        result = data
        
        for i, processor in enumerate(self.processors):
            try:
                result = processor.process(result)
                logger.debug(f"处理器 {i+1}/{len(self.processors)} 完成")
            except Exception as e:
                logger.error(f"处理器 {i+1} 执行失败: {e}")
                raise
        
        return result
    
    def batch_process(
        self, 
        data_list: List[Any], 
        show_progress: bool = False
    ) -> List[Any]:
        """
        批量处理数据
        
        Args:
            data_list: 输入数据列表
            show_progress: 是否显示进度
            
        Returns:
            处理后的数据列表
        """
        results = []
        total = len(data_list)
        
        for i, data in enumerate(data_list):
            if show_progress and (i + 1) % 100 == 0:
                logger.info(f"处理进度: {i+1}/{total}")
            
            try:
                result = self.process(data)
                results.append(result)
            except Exception as e:
                logger.error(f"处理第 {i+1} 项数据失败: {e}")
                results.append(None)
        
        return results
    
    def clear(self) -> None:
        """清空所有处理器"""
        self.processors.clear()
    
    def __len__(self) -> int:
        """返回处理器数量"""
        return len(self.processors)


class DataValidator:
    """
    数据验证器
    提供各种数据验证功能
    """
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """验证邮箱格式"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def is_valid_phone(phone: str) -> bool:
        """验证手机号格式（中国）"""
        pattern = r'^1[3-9]\d{9}$'
        return bool(re.match(pattern, phone))
    
    @staticmethod
    def is_valid_id_card(id_card: str) -> bool:
        """验证身份证号格式（中国）"""
        pattern = r'^\d{17}[\dXx]$'
        return bool(re.match(pattern, id_card))
    
    @staticmethod
    def is_valid_age(age: int, min_age: int = 0, max_age: int = 150) -> bool:
        """验证年龄范围"""
        return min_age <= age <= max_age
    
    @staticmethod
    def is_not_empty(value: Any) -> bool:
        """验证值非空"""
        if value is None:
            return False
        if isinstance(value, (str, list, dict)) and len(value) == 0:
            return False
        return True
    
    @staticmethod
    def is_in_range(value: float, min_val: float, max_val: float) -> bool:
        """验证值在指定范围内"""
        return min_val <= value <= max_val
    
    @staticmethod
    def validate_dict_fields(
        data: Dict[str, Any], 
        required_fields: List[str],
        optional_fields: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        验证字典字段
        
        Args:
            data: 要验证的字典
            required_fields: 必需字段列表
            optional_fields: 可选字段列表
            
        Returns:
            (是否通过验证, 错误信息列表)
        """
        errors = []
        
        # 检查必需字段
        for field in required_fields:
            if field not in data:
                errors.append(f"缺少必需字段: {field}")
            elif not DataValidator.is_not_empty(data[field]):
                errors.append(f"字段不能为空: {field}")
        
        # 检查未知字段（如果提供了optional_fields）
        if optional_fields is not None:
            all_valid_fields = set(required_fields + optional_fields)
            unknown_fields = set(data.keys()) - all_valid_fields
            if unknown_fields:
                errors.append(f"未知字段: {unknown_fields}")
        
        return len(errors) == 0, errors