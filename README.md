# 医疗推荐系统

## 一、快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行演示

```bash
python scripts/run_agent_demo.py
```

### 3. 运行测试

```bash
python scripts/test_basic_functions.py
```

## 二、核心功能

### 1. 诊断服务

- 基于症状的疾病诊断
- 鉴别诊断
- 风险评估

### 2. 药品推荐

- 个性化药品推荐
- 药物相互作用检查
- 过敏史筛查
- 用药安全评估

### 3. 治疗方案

- 分阶段治疗计划
- 生活方式建议
- 复诊计划制定
- 预期结果评估

### 4. 医疗问答

- 智能问答系统
- 知识检索
- 相关问题推荐

### 5. 智能 Agent

- 多工具协作
- 自主决策
- 上下文管理
- 行动记录

## 三、使用示例

### 基础问答

```python
from medical_recommendation.core import RecommendationEngine, RecommendationRequest

# 创建引擎
engine = RecommendationEngine(use_mock=True)

# 问答请求
request = RecommendationRequest(
    request_type="qa",
    query="高血压有哪些症状？"
)
response = engine.recommend(request)
print(response.result['answer'])
```

### 症状诊断

```python
request = RecommendationRequest(
    request_type="diagnosis",
    patient_id="P001",
    query="症状诊断",
    context={
        "symptoms": ["头痛", "发热", "咳嗽"],
        "duration": "3天",
        "severity": "moderate"
    }
)
response = engine.recommend(request)
print(f"诊断: {response.result['primary_diagnosis']['disease_name']}")
```

### Agent 咨询

```python
from medical_recommendation.agents import MedicalAgent, AgentInput, AgentMode

agent = MedicalAgent(use_mock=True)
input_data = AgentInput(
    query="我最近总是头痛，可能是什么原因？",
    mode=AgentMode.CONSULTATION
)
output = agent.run(input_data)
print(output.answer)
```

## 四、系统架构

```
用户请求
    ↓
推荐引擎 (RecommendationEngine)
    ↓
┌───────────────┬──────────────┬────────────────┐
│ 诊断链        │ 药品推荐链   │ 治疗方案链     │
└───────────────┴──────────────┴────────────────┘
    ↓
混合检索器 (HybridRetriever)
    ↓
┌──────────────┬─────────────────┐
│ 知识图谱检索 │ 向量相似度检索  │
└──────────────┴─────────────────┘
    ↓
结果融合与返回
```

## 五、模块说明

### Core 模块

- `PatientProfileManager`: 患者画像构建和管理
- `KnowledgeGraphManager`: 知识图谱查询操作
- `VectorStoreManager`: 向量存储管理
- `HybridRetriever`: 混合检索策略
- `LLMChainManager`: LLM 调用封装
- `RecommendationEngine`: 推荐引擎主逻辑

### Chains 模块

- `DiagnosisChain`: 诊断处理链
- `DrugRecommendationChain`: 药品推荐链
- `TreatmentPlanChain`: 治疗方案链
- `QAChain`: 问答处理链

### Agents 模块

- `MedicalAgent`: 医疗顾问智能体
- `KGQueryTool`: 知识图谱查询工具
- `VectorSearchTool`: 向量搜索工具
- `PatientInfoTool`: 患者信息工具

### Retrievers 模块

- `KGRetriever`: 知识图谱检索器
- `VectorRetriever`: 向量检索器
- `EnsembleRetriever`: 集成检索器

## 六、配置

系统使用 YAML 配置文件，位于 `config/` 目录：

- `llm_config.yaml`: LLM 模型配置
- `vector_config.yaml`: 向量存储配置
- `kg_config.yaml`: 知识图谱配置
- `chain_config.yaml`: 链配置
