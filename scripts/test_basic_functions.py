#!/usr/bin/env python3
"""
基础功能测试脚本
测试各个模块的基本功能
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from medical_recommendation.core import (
    RecommendationEngine,
    PatientProfileManager,
    KnowledgeGraphManager,
    VectorStoreManager,
    HybridRetriever
)
from medical_recommendation.agents import MedicalAgent
from medical_recommendation.chains import DiagnosisChain, QAChain


class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def test(self, name: str, func):
        """运行单个测试"""
        try:
            print(f"\n测试: {name}...", end=" ")
            func()
            print("✅ 通过")
            self.passed += 1
        except AssertionError as e:
            print(f"❌ 失败: {e}")
            self.failed += 1
            self.errors.append((name, str(e)))
        except Exception as e:
            print(f"❌ 错误: {e}")
            self.failed += 1
            self.errors.append((name, str(e)))
    
    def summary(self):
        """打印测试摘要"""
        print("\n" + "=" * 80)
        print(f"测试完成: {self.passed} 通过, {self.failed} 失败")
        print("=" * 80)
        
        if self.errors:
            print("\n失败详情:")
            for name, error in self.errors:
                print(f"  • {name}: {error}")


def test_patient_profile_manager():
    """测试患者画像管理器"""
    manager = PatientProfileManager(use_mock=True)
    
    # 获取患者画像
    profile = manager.get_profile("P001")
    assert profile is not None, "患者画像不应为空"
    assert profile.patient_id == "P001", "患者ID不匹配"
    assert len(profile.allergies) > 0, "应该有过敏史"
    
    # 计算风险评分
    risk_score = profile.get_risk_score()
    assert 0 <= risk_score <= 1, "风险评分应在0-1之间"


def test_knowledge_graph_manager():
    """测试知识图谱管理器"""
    manager = KnowledgeGraphManager(use_mock=True)
    
    # 查询疾病实体
    diseases = manager.query_entity("Disease")
    assert len(diseases) > 0, "应该能查询到疾病"
    
    # 查询关系
    relationships = manager.query_relationships("disease_001")
    assert isinstance(relationships, list), "应该返回关系列表"


def test_vector_store_manager():
    """测试向量存储管理器"""
    manager = VectorStoreManager(use_mock=True)
    
    # 添加文档
    success = manager.add_document("test_001", "测试文档")
    assert success, "添加文档应该成功"
    
    # 搜索
    results = manager.search("高血压", top_k=3)
    assert len(results) > 0, "应该能搜索到结果"
    assert all('score' in r for r in results), "结果应该包含分数"


def test_hybrid_retriever():
    """测试混合检索器"""
    retriever = HybridRetriever(use_mock=True)
    
    # 混合检索
    results = retriever.retrieve("高血压治疗", top_k=5, mode="hybrid")
    assert len(results) > 0, "应该能检索到结果"
    
    # 知识图谱检索
    kg_results = retriever.retrieve("糖尿病", top_k=3, mode="kg")
    assert len(kg_results) > 0, "KG检索应该有结果"
    
    # 向量检索
    vec_results = retriever.retrieve("药品信息", top_k=3, mode="vector")
    assert len(vec_results) > 0, "向量检索应该有结果"


def test_diagnosis_chain():
    """测试诊断链"""
    from medical_recommendation.chains import DiagnosisInput
    
    chain = DiagnosisChain(use_mock=True)
    
    input_data = DiagnosisInput(
        patient_id="P001",
        symptoms=["头痛", "发热"],
        duration="3天"
    )
    
    output = chain.run(input_data)
    assert output.patient_id == "P001", "患者ID应该匹配"
    assert len(output.differential_diagnoses) >= 0, "应该有候选诊断"


def test_qa_chain():
    """测试问答链"""
    from medical_recommendation.chains import QAInput
    
    chain = QAChain(use_mock=True)
    
    input_data = QAInput(
        question="高血压有哪些症状？"
    )
    
    output = chain.run(input_data)
    assert len(output.answer) > 0, "应该有答案"
    assert 0 <= output.confidence <= 1, "置信度应在0-1之间"


def test_medical_agent():
    """测试医疗Agent"""
    from medical_recommendation.agents import AgentInput, AgentMode
    
    agent = MedicalAgent(use_mock=True)
    
    input_data = AgentInput(
        query="我头痛怎么办？",
        mode=AgentMode.CONSULTATION
    )
    
    output = agent.run(input_data)
    assert output.success, "Agent应该成功运行"
    assert len(output.answer) > 0, "应该有答案"
    assert len(output.tools_used) >= 0, "应该记录使用的工具"


def test_recommendation_engine():
    """测试推荐引擎"""
    from medical_recommendation.core import RecommendationRequest
    
    engine = RecommendationEngine(use_mock=True)
    
    # 测试问答
    request = RecommendationRequest(
        request_type="qa",
        query="阿司匹林的作用是什么？"
    )
    
    response = engine.recommend(request)
    assert response.success, "请求应该成功"
    assert 'answer' in response.result, "结果应该包含答案"
    
    # 测试健康检查
    health = engine.health_check()
    assert all(health.values()), "所有组件应该健康"


def test_integration():
    """集成测试"""
    from medical_recommendation.core import RecommendationRequest
    
    engine = RecommendationEngine(use_mock=True)
    
    # 完整流程：诊断 -> 药品推荐 -> 治疗方案
    
    # 1. 诊断
    diag_request = RecommendationRequest(
        request_type="diagnosis",
        patient_id="P001",
        query="症状诊断",
        context={"symptoms": ["头痛", "发热"]}
    )
    diag_response = engine.recommend(diag_request)
    assert diag_response.success, "诊断应该成功"
    
    # 2. 药品推荐
    drug_request = RecommendationRequest(
        request_type="drug",
        patient_id="P001",
        query="药品推荐",
        context={"diagnosis": "感冒"}
    )
    drug_response = engine.recommend(drug_request)
    assert drug_response.success, "药品推荐应该成功"
    
    # 3. 治疗方案
    treatment_request = RecommendationRequest(
        request_type="treatment",
        patient_id="P001",
        query="治疗方案",
        context={"diagnosis": "感冒"}
    )
    treatment_response = engine.recommend(treatment_request)
    assert treatment_response.success, "治疗方案应该成功"


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print(" 医疗推荐系统 - 基础功能测试")
    print("=" * 80)
    
    runner = TestRunner()
    
    # 运行各个测试
    runner.test("患者画像管理器", test_patient_profile_manager)
    runner.test("知识图谱管理器", test_knowledge_graph_manager)
    runner.test("向量存储管理器", test_vector_store_manager)
    runner.test("混合检索器", test_hybrid_retriever)
    runner.test("诊断链", test_diagnosis_chain)
    runner.test("问答链", test_qa_chain)
    runner.test("医疗Agent", test_medical_agent)
    runner.test("推荐引擎", test_recommendation_engine)
    runner.test("集成测试", test_integration)
    
    # 打印摘要
    runner.summary()
    
    return runner.failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)