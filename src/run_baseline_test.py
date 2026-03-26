import pandas as pd
import time
import random
# 导入你之前写好的 agent_demo 中的核心类
from agent_demo import DualStreamAgentSystem


def generate_baseline_logs(excel_path='../data/goods_batch_1_20260130_180359.xlsx', test_sample_size=10,
                           build_full_kb=True):
    """
    全自动跑一遍电商数据，收集原始的 QA 日志，用于后续的人工审核和微调冷启动。
    包含知识库的构建与自动化基线测试。
    """
    print("🤖 正在启动自动化测试流水线 ...")

    # 初始化你的智能客服系统
    agent_system = DualStreamAgentSystem(db_path="../agent_final_db")

    # ==========================================
    # 第一步：构建/同步完整的知识库
    # ==========================================
    if build_full_kb:
        print("\n📚 [阶段一] 正在构建/同步完整的商品知识库...")
        df_full = pd.read_excel(excel_path)

        # 将完整的数据丢入知识库构建管道（已有去重/覆盖机制，可放心运行）
        agent_system.build_knowledge_base(df_full)

        # 打印一下知识库最新状态
        agent_system.get_database_stats()
        print("✅ 知识库构建完毕！\n")

    # ==========================================
    # 第二步：自动化基线测试 (Baseline Evaluation)
    # ==========================================
    print("🧪 [阶段二] 正在进行自动化基线测试...")
    # 只抽取部分商品进行提问测试，避免 API 消耗过大
    df_test = pd.read_excel(excel_path).head(test_sample_size)

    success_count = 0

    for index, row in df_test.iterrows():
        model = str(row.get('商品型号', ''))
        name = str(row.get('商品完整名称', ''))

        if not model or model.lower() == 'nan':
            continue

        # 模拟用户提问（可以随机变换几种问法）
        queries = [
            f"你好，请问 {model} 这个型号有什么参数？",
            f"我想了解一下 {name}，价格是多少？",
            f"帮我查一下 {model}，这东西不错，给我下单 {random.randint(10, 100)} 个。"
        ]

        # 随机挑一个问题问大模型
        user_query = random.choice(queries)

        print(f"\n--- 正在测试第 {index + 1}/{len(df_test)} 个商品 ---")
        print(f"👤 虚拟用户提问: {user_query}")

        # 调用你的客服系统 (这里会自动触发 qa_feedback_logs.jsonl 的写入)
        reply, order, history, i_id = agent_system.chat_with_agent(user_query)

        print(f"🤖 模型原始回答: {reply}")
        if order:
            print(f"📦 成功截获订单: {order}")

        success_count += 1

        # 稍微休眠一下，防止触发大模型 API 的高频限制
        time.sleep(2)

    print(f"\n🎉 基线测试流水线完成！共生成 {success_count} 条真实的原始交互日志。")
    print("请打开 qa_feedback_logs.jsonl 进行人工评估与修正。")


if __name__ == "__main__":
    # 如果你想跳过建库直接测试，可以将 build_full_kb 设为 False
    generate_baseline_logs(build_full_kb=False, test_sample_size=10)