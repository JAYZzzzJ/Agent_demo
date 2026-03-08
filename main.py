# 1-8   前七个文件 5w条 8
# brand_id_name 关联名牌表
# jdgph_category 关联分类表


import pandas as pd
import json
import difflib


# ==========================================
# 1. 模拟组件：OCR 与 数据库
# ==========================================

# 模拟 OCR (光学字符识别) 模块
# 在实际项目中，这里应调用 PaddleOCR / Tesseract / Google Vision API
def mock_ocr_process(image_urls):
    if pd.isna(image_urls) or image_urls == "":
        return ""

    # 模拟：假设从图片中提取到了额外的技术参数
    # 真实代码应下载图片 -> 运行OCR -> 返回文本
    # 这里我们返回一个通用的占位符，演示数据流
    return " [图片提取信息: 包含详细接线图，额定绝缘电压600V，符合IEC60947标准] "


# 模拟 品牌ID 映射表 (实际应读取单独的数据库表)
BRAND_MAP = {
    97: "施耐德电气 (Schneider)",
    95: "ABB",
    96: "西门子 (Siemens)",
    225: "和泉 (IDEC)",
    210: "伊顿穆勒 (Eaton)",
    205: "APT",
    103: "欧姆龙 (Omron)",
    104: "德力西 (Delixi)"
}


# ==========================================
# 2. 核心类：电商知识库构建器
# ==========================================

class ECommerceKnowledgeBase:
    def __init__(self, csv_path):
        self.products = []
        self.load_data(csv_path)

    def load_data(self, csv_path):
        print("正在加载并处理数据...")
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            # A. 处理基础信息
            brand_name = BRAND_MAP.get(row['品牌ID'], f"品牌ID_{row['品牌ID']}")

            # B. 处理扩展属性 (JSON解析)
            try:
                attrs = json.loads(row['扩展属性'])
                # 将字典转换为自然语言描述
                attr_desc = ", ".join([f"{k}是{v}" for k, v in attrs.items()])
            except:
                attr_desc = "暂无详细属性"

            # C. 处理商详图片 (OCR提取)
            # 这里调用上面的模拟OCR函数，将提取的文字合并到知识中
            img_text = mock_ocr_process(row['商详图片'])

            # D. 构建完整的“知识块” (Knowledge Chunk)
            # 这是喂给大模型的核心内容
            full_description = (
                f"商品名称: {row['商品完整名称']}\n"
                f"品牌: {brand_name}\n"
                f"型号: {row['商品型号']}\n"
                f"价格: {row['当前实际销售价格']}元\n"
                f"参数详情: {attr_desc}\n"
                f"图片补充信息: {img_text}"
            )

            self.products.append({
                "id": row['id'],
                "model": str(row['商品型号']),
                "name": row['商品完整名称'],
                "price": row['当前实际销售价格'],
                "knowledge_text": full_description
            })
        print(f"数据加载完成，共处理 {len(self.products)} 个SKU。")

    # 模拟检索功能 (Retrieval)
    # 实际项目中应使用 Vector DB (如 Milvus, Chroma) 进行语义搜索
    # 这里用简单的关键词匹配演示
    def search(self, query):
        results = []
        query_parts = query.lower().split()
        for p in self.products:
            score = 0
            # 简单的加权规则
            if query_parts[0] in p['model'].lower(): score += 5
            if query_parts[0] in p['name'].lower(): score += 3
            if query_parts[0] in p['knowledge_text'].lower(): score += 1

            if score > 0:
                results.append(p)

        # 按相关度排序返回前3个
        return sorted(results, key=lambda x: x['id'], reverse=True)[:3]

    # 精确查找（用于下单）
    def find_by_model(self, model_no):
        for p in self.products:
            if p['model'] == model_no:
                return p
        return None


# ==========================================
# 3. 核心类：智能客服 Agent
# ==========================================

class CustomerServiceAgent:
    def __init__(self, kb):
        self.kb = kb
        self.history = []  # 对话历史，用于RLHF数据收集

    def generate_answer(self, user_query):
        # 1. 检索阶段
        retrieved_items = self.kb.search(user_query)

        if not retrieved_items:
            return "抱歉，我没有找到相关的电子产品信息。请提供更准确的型号或参数。"

        # 2. 增强生成阶段 (Prompt Engineering)
        # 将检索到的知识注入Prompt
        context = "\n---\n".join([item['knowledge_text'] for item in retrieved_items])

        # 模拟 LLM 生成回答 (实际应调用 GPT-4 或 DeepSeek 等 API)
        # 这里用规则模板模拟 LLM 的输出风格
        answer = f"""
[AI 客服]: 基于您的需求，我为您找到了以下产品：

{retrieved_items[0]['knowledge_text']}

针对您的问题：这些产品通常用于工业控制或电路板设计。其中 {retrieved_items[0]['model']} 是非常畅销的型号，它包含{retrieved_items[0]['price']}元的价格优势。
如果您是海外客户，请注意该产品的额定电压参数是否符合您当地标准。
"""
        # 3. 记录日志 (为RLHF做准备)
        self.log_interaction(user_query, answer)
        return answer

    def create_order(self, model_no, quantity):
        product = self.kb.find_by_model(model_no)
        if not product:
            return json.dumps({"error": "Product not found"}, ensure_ascii=False)

        total_price = product['price'] * quantity

        # 生成订单 JSON
        order_info = {
            "order_id": "ORD-20260211-001",
            "items": [
                {
                    "product_name": product['name'],
                    "model": model_no,
                    "unit_price": product['price'],
                    "quantity": quantity,
                    "total": round(total_price, 2)
                }
            ],
            "currency": "CNY",
            "status": "pending_payment"
        }
        return json.dumps(order_info, indent=4, ensure_ascii=False)

    def log_interaction(self, query, answer):
        # 在实际系统中，这里会将数据写入数据库，并提供给后台让人工打分
        self.history.append({"q": query, "a": answer, "user_rating": None})


# ==========================================
# 4. 运行 Demo
# ==========================================

# 初始化知识库
csv_file = "goods_batch_1_20260130_180359.xlsx - 商品数据.csv"
kb = ECommerceKnowledgeBase(csv_file)
agent = CustomerServiceAgent(kb)

# 场景 1: 外国客户咨询 (模拟模糊提问)
print("--- 场景 1: 客户咨询 ---")
user_question = "XD2PA22CR"
print(f"用户提问: {user_question}")
response = agent.generate_answer(user_question)
print(response)

# 场景 2: 客户确认下单
print("\n--- 场景 2: 生成订单 ---")
order_model = "XD2PA22CR"
quantity = 50
print(f"用户指令: 我要订购 50 个 {order_model}")
order_json = agent.create_order(order_model, quantity)
print("系统生成的订单信息:")
print(order_json)