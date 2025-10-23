#!/usr/bin/env python3
"""
医疗推荐系统 - Agent演示脚本
展示系统的核心功能
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from medical_recommendation.core import RecommendationEngine, RecommendationRequest
from medical_recommendation.agents import MedicalAgent, AgentInput, AgentMode
import json


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def demo_basic_qa():
    """演示基础问答功能"""
    print_section("演示1: 基础医疗问答")
    
    # 创建推荐引擎
    engine = RecommendationEngine(use_mock=True)
    
    # 问答请求
    request = RecommendationRequest(
        request_type="qa",
        query="高血压有哪些症状？"
    )
    
    response = engine.recommend(request)
    
    print(f"问题: {request.query}")
    print(f"答案:\n{response.result.get('answer', '')}")
    print(f"置信度: {response.confidence:.2%}")
    
    if response.result.get('related_questions'):
        print("\n相关问题:")
        for q in response.result['related_questions']:
            print(f"  • {q}")


def demo_diagnosis():
    """演示诊断功能"""
    print_section("演示2: 症状诊断")
    
    engine = RecommendationEngine(use_mock=True)
    
    # 诊断请求
    request = RecommendationRequest(
        request_type="diagnosis",
        patient_id="P001",
        query="患者症状分析",
        context={
            "symptoms": ["头痛", "发热", "咳嗽"],
            "duration": "3天",
            "severity": "moderate"
        }
    )
    
    response = engine.recommend(request)
    
    print(f"症状: {', '.join(request.context['symptoms'])}")
    print(f"持续时间: {request.context['duration']}")
    
    if response.result.get('primary_diagnosis'):
        diag = response.result['primary_diagnosis']
        print(f"\n主要诊断: {diag['disease_name']}")
        print(f"置信度: {diag['confidence']:.2%}")
        print(f"推理: {diag['reasoning']}")
    
    if response.recommendations:
        print("\n建议:")
        for rec in response.recommendations:
            print(f"  ✓ {rec}")


def demo_drug_recommendation():
    """演示药品推荐功能"""
    print_section("演示3: 药品推荐")
    
    engine = RecommendationEngine(use_mock=True)
    
    # 药品推荐请求
    request = RecommendationRequest(
        request_type="drug",
        patient_id="P001",
        query="急性上呼吸道感染的用药",
        context={
            "diagnosis": "急性上呼吸道感染（感冒）",
            "symptoms": ["发热", "咳嗽"]
        }
    )
    
    response = engine.recommend(request)
    
    print(f"诊断: {request.context['diagnosis']}")
    
    if response.result.get('primary_drugs'):
        print("\n主要推荐药品:")
        for i, drug in enumerate(response.result['primary_drugs'][:3], 1):
            print(f"\n{i}. {drug['drug_name']}")
            print(f"   用法: {drug['dosage']}, {drug['frequency']}")
            print(f"   疗程: {drug.get('duration', '遵医嘱')}")
            print(f"   置信度: {drug['confidence']:.2%}")
    
    if response.warnings:
        print("\n⚠️  警告:")
        for warning in response.warnings:
            print(f"  {warning}")


def demo_treatment_plan():
    """演示治疗方案功能"""
    print_section("演示4: 治疗方案制定")
    
    engine = RecommendationEngine(use_mock=True)
    
    # 治疗方案请求
    request = RecommendationRequest(
        request_type="treatment",
        patient_id="P001",
        query="高血压治疗方案",
        context={
            "diagnosis": "高血压",
            "severity": "moderate"
        }
    )
    
    response = engine.recommend(request)
    
    print(f"诊断: {request.context['diagnosis']}")
    print(f"严重程度: {request.context['severity']}")
    
    if response.result.get('strategy'):
        print(f"\n治疗策略:\n{response.result['strategy']}")
    
    if response.result.get('phases'):
        print("\n治疗阶段:")
        for i, phase in enumerate(response.result['phases'], 1):
            print(f"\n阶段{i}: {phase['phase_name']}")
            print(f"  时长: {phase['duration']}")
            print(f"  目标: {', '.join(phase['goals'])}")
    
    if response.result.get('duration'):
        print(f"\n预计治疗时长: {response.result['duration']}")


def demo_agent_consultation():
    """演示Agent咨询功能"""
    print_section("演示5: 智能Agent咨询")
    
    # 创建Agent
    agent = MedicalAgent(use_mock=True)
    
    # 咨询场景
    queries = [
        "我最近总是头痛，可能是什么原因？",
        "阿司匹林有什么副作用？",
        "如何预防糖尿病？"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- 咨询 {i} ---")
        print(f"问题: {query}")
        
        agent_input = AgentInput(
            query=query,
            mode=AgentMode.CONSULTATION
        )
        
        output = agent.run(agent_input)
        
        print(f"\n答案:\n{output.answer}")
        print(f"\n使用的工具: {', '.join(output.tools_used)}")
        print(f"执行步骤: {len(output.actions)} 步")
        
        if i < len(queries):
            print("\n" + "-" * 60)


def demo_patient_profile():
    """演示患者画像功能"""
    print_section("演示6: 患者画像管理")
    
    engine = RecommendationEngine(use_mock=True)
    
    # 获取患者画像
    patient_id = "P001"
    profile = engine.get_patient_profile(patient_id)
    
    if profile:
        print(f"患者ID: {patient_id}")
        print(f"基本信息: {profile.get('basic_info', {})}")
        print(f"慢性病: {', '.join(profile.get('chronic_conditions', []))}")
        print(f"过敏史: {', '.join(profile.get('allergies', []))}")
        print(f"风险因素: {', '.join(profile.get('risk_factors', []))}")
        
        # 计算风险评分
        from medical_recommendation.core import PatientProfile
        prof_obj = PatientProfile(**profile)
        risk_score = prof_obj.get_risk_score()
        print(f"\n综合风险评分: {risk_score:.2f} / 1.00")


def demo_knowledge_search():
    """演示知识搜索功能"""
    print_section("演示7: 医疗知识搜索")
    
    engine = RecommendationEngine(use_mock=True)
    
    queries = [
        "高血压的治疗方法",
        "糖尿病的饮食建议",
        "阿司匹林的作用机制"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        results = engine.search_knowledge(query, top_k=3, mode="hybrid")
        
        print(f"找到 {len(results)} 条结果:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [类型: {result['type']}] (相关度: {result['score']:.2%})")
            print(f"   {result['content'][:100]}...")


def demo_system_info():
    """演示系统信息查询"""
    print_section("演示8: 系统信息")
    
    engine = RecommendationEngine(use_mock=True)
    
    # 健康检查
    health = engine.health_check()
    print("系统健康状态:")
    for component, status in health.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component}: {'正常' if status else '异常'}")
    
    # 统计信息
    stats = engine.get_statistics()
    print("\n系统统计:")
    print(f"  知识图谱: {json.dumps(stats.get('knowledge_graph', {}), ensure_ascii=False, indent=4)}")
    print(f"  向量存储: {json.dumps(stats.get('vector_store', {}), ensure_ascii=False, indent=4)}")
    print(f"  患者画像缓存: {stats.get('patient_profiles', 0)} 个")
    print(f"  系统状态: {stats.get('system_status', 'unknown')}")


def interactive_mode():
    """交互式模式"""
    print_section("交互式咨询模式")
    
    print("欢迎使用医疗推荐系统!")
    print("提示: 输入 'quit' 或 'exit' 退出")
    print("      输入 'help' 查看帮助")
    
    engine = RecommendationEngine(use_mock=True)
    
    while True:
        print("\n" + "-" * 60)
        query = input("\n请输入您的问题: ").strip()
        
        if not query:
            continue
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n感谢使用，再见！")
            break
        
        if query.lower() == 'help':
            print("\n可用命令:")
            print("  - 直接输入问题进行咨询")
            print("  - 'quit' 或 'exit': 退出")
            print("  - 'help': 显示帮助")
            print("\n示例问题:")
            print("  • 高血压有哪些症状？")
            print("  • 阿司匹林的副作用是什么？")
            print("  • 如何预防糖尿病？")
            continue
        
        # 处理问题
        request = RecommendationRequest(
            request_type="qa",
            query=query
        )
        
        try:
            response = engine.recommend(request)
            
            if response.success:
                print(f"\n回答:\n{response.result.get('answer', '')}")
                
                if response.recommendations:
                    print("\n建议:")
                    for rec in response.recommendations:
                        print(f"  • {rec}")
            else:
                print(f"\n处理失败: {response.error_message}")
        
        except Exception as e:
            print(f"\n发生错误: {e}")


def main():
    """主函数"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "医疗推荐系统 - 功能演示" + " " * 20 + "║")
    print("╚" + "═" * 78 + "╝")
    
    demos = {
        "1": ("基础问答", demo_basic_qa),
        "2": ("症状诊断", demo_diagnosis),
        "3": ("药品推荐", demo_drug_recommendation),
        "4": ("治疗方案", demo_treatment_plan),
        "5": ("Agent咨询", demo_agent_consultation),
        "6": ("患者画像", demo_patient_profile),
        "7": ("知识搜索", demo_knowledge_search),
        "8": ("系统信息", demo_system_info),
        "9": ("交互模式", interactive_mode),
        "0": ("运行全部", None)
    }
    
    while True:
        print("\n" + "=" * 80)
        print("请选择演示项目:")
        print("=" * 80)
        
        for key, (name, _) in demos.items():
            print(f"  [{key}] {name}")
        
        print("\n  [q] 退出程序")
        
        choice = input("\n请输入选项 [0-9/q]: ").strip().lower()
        
        if choice == 'q':
            print("\n感谢使用！")
            break
        
        if choice == '0':
            # 运行所有演示
            for key in ['1', '2', '3', '4', '5', '6', '7', '8']:
                try:
                    _, demo_func = demos[key]
                    if demo_func:
                        demo_func()
                        input("\n按回车键继续...")
                except Exception as e:
                    print(f"\n演示出错: {e}")
                    import traceback
                    traceback.print_exc()
        elif choice in demos and choice != '0':
            name, demo_func = demos[choice]
            if demo_func:
                try:
                    demo_func()
                    if choice != '9':  # 交互模式不需要暂停
                        input("\n按回车键返回主菜单...")
                except Exception as e:
                    print(f"\n演示出错: {e}")
                    import traceback
                    traceback.print_exc()
                    input("\n按回车键返回主菜单...")
        else:
            print("\n无效选项，请重新选择！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断，退出中...")
    except Exception as e:
        print(f"\n程序发生错误: {e}")
        import traceback
        traceback.print_exc()