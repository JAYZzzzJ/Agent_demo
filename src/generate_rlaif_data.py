import json
import uuid
from datetime import datetime
import os
import pandas as pd
import dashscope
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(override=True)
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")


def generate_rlaif_dpo_logs(excel_path="../data/goods_batch_1_20260130_180359.xlsx", filename="qa_feedback_logs.jsonl",
                            sample_size=5):
    """
    全自动构造 RLAIF 偏好数据集，结合真实商品数据与 Qwen-Max 专家模型。
    """
    # 1. 动态读取真实的业务数据
    if not os.path.exists(excel_path):
        print(f"❌ 找不到真实数据文件: {excel_path}，请检查路径。")
        return

    df = pd.read_excel(excel_path).head(sample_size)
    logs = []

    print(f"🚀 开始读取真实数据，并调用 Qwen-Max (Teacher Model) 生成 RLAIF 偏好对...")

    for index, row in df.iterrows():
        interaction_id = str(uuid.uuid4())

        # 提取真实商品属性
        model = str(row.get('商品型号', '未知型号'))
        price = row.get('当前实际销售价格', 0)
        name = str(row.get('商品完整名称', '未知名称'))
        qty = 50  # 设定一个测试购买数量

        # 构建真实的检索上下文
        context = f"【基础信息】商品名称：{name}，型号：{model}\n【价格】{price}元"
        user_query = f"这价格不错，给我来 {qty} 个 {model} 吧，急用。"

        # 2. 模拟“学生模型”的坏回答 (Rejected)
        # 也就是模型原本容易犯的错：态度很好，但忘记输出结构化的 JSON 订单
        bad_model_response = f"好的，没问题！{name} 这款产品非常适合您，{qty} 个的总价是 {qty * float(price)} 元。请问您需要寄到哪里？"

        # 3. 让“专家模型”(Qwen-Max) 自动生成好回答 (Chosen)
        # 这就是 RLAIF 的核心：用大模型来矫正大模型
        system_prompt = f"""
        你是一个高级AI客服主管。你的下属在面对客户下单时，忘记输出后台需要的 JSON 订单代码了。
        请你示范一个完美的回答。
        要求：
        1. 态度热情专业，确认商品型号 {model} 和总价 {qty * float(price)} 元。
        2. 必须在回复末尾严格附带以下 JSON，用于系统拦截：
        ```json
        {{
            "intent": "create_order",
            "model": "{model}",
            "quantity": {qty},
            "unit_price": {price}
        }}
        ```
        """

        try:
            # 调用百炼平台的顶级模型充当“人类专家”
            response = dashscope.Generation.call(
                model='qwen-max',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user',
                     'content': f"客户说：{user_query}\n下属的糟糕回答是：{bad_model_response}\n请你给出完美示范回答："}
                ],
                result_format='message'
            )
            good_ai_response = response.output.choices[0]['message']['content']
        except Exception as e:
            print(f"⚠️ 专家模型调用失败，跳过该条: {e}")
            continue

        # 4. 组装为 DPO 日志格式
        log_entry = {
            "interaction_id": interaction_id,
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "user_query": user_query,
            "model_response": bad_model_response,
            "human_corrected_response": good_ai_response,  # AI代劳的人类修正
            "rating": -1  # 标记为负面，激活微调机制
        }
        logs.append(log_entry)
        print(f"✅ 成功生成 1 条基于真实商品 [{model}] 的 RLAIF 偏好数据。")

    # 5. 写入日志文件
    mode = 'a' if os.path.exists(filename) else 'w'
    with open(filename, mode, encoding='utf-8') as f:
        for log in logs:
            f.write(json.dumps(log, ensure_ascii=False) + '\n')

    print(f"🎉 成功生成 {len(logs)} 条结合真实数据的 RLAIF 高质量偏好数据，并已追加到 {filename}！")
    print("现在你可以直接运行 Mac 专属的 DPO 微调脚本了。")


if __name__ == "__main__":
    # 如果你的数据路径不同，请在这里修改 excel_path
    generate_rlaif_dpo_logs()