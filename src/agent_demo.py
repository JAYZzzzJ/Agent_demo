import base64
import os
import pandas as pd
import json
import chromadb
from PIL import Image
from io import BytesIO
import requests
import dashscope
from sentence_transformers import SentenceTransformer
import time
import re
import uuid
from datetime import datetime
from dotenv import load_dotenv

# ==========================================
# 0. 环境与 API 密钥配置
# ==========================================
load_dotenv(override=True)

QWEN_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "").strip()
dashscope.api_key = QWEN_API_KEY
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'


# ==========================================
# 工具函数：图片下载与拼接
# ==========================================
def download_image_with_auth(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://www.jd.com/'
    }
    try:
        resp = requests.get(url.strip(), headers=headers, timeout=10)
        if resp.status_code == 200:
            return Image.open(BytesIO(resp.content)).convert('RGB')
        return None
    except Exception:
        return None


def stitch_images_vertically(image_urls):
    images = []
    for url in image_urls:
        if not url.strip(): continue
        img = download_image_with_auth(url)
        if img: images.append(img)
    if not images: return None

    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights)

    stitched_img = Image.new('RGB', (max_width, total_height), color=(255, 255, 255))
    y_offset = 0
    for im in images:
        stitched_img.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return stitched_img


# ==========================================
# 1. 图像处理与视觉理解 (Qwen-VL)
# ==========================================
def extract_specs_from_stitched_image(stitched_img):
    """使用 Qwen-VL 提取纯文本参数"""
    if stitched_img is None: return ""

    buffered = BytesIO()
    stitched_img.save(buffered, format="JPEG", quality=80)  # 加上 quality 稍微压缩
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{img_base64}"

    # 【修改标注 1】：优化了提示词，让模型积极总结哪怕不是表格的“说明性”特征
    prompt = """
    你是一个资深的电气自动化工程师。这是一张产品商详图。
    请仔细分析图中的技术图纸、尺寸标注和产品说明。
    请用清晰的中文，罗列出所有有价值的技术参数（如电压、孔径、材质等）。
    即使没有标准的参数表格，只要图中包含了产品的核心特征、尺寸（如60mm）或适用场景（如急停标牌），也请务必提取并总结出来！
    只有在完全没有任何产品说明和尺寸信息时，才回复“无附加图纸参数”。
    """
    try:
        messages = [{"role": "user", "content": [{"image": image_url}, {"text": prompt}]}]
        response = dashscope.MultiModalConversation.call(
            model='qwen-vl-max',
            messages=messages,
            result_format='message'
        )
        if response.status_code == 200:
            raw_content = response.output.choices[0]['message']['content']
            result_text = ""
            if isinstance(raw_content, list):
                for item in raw_content:
                    if 'text' in item: result_text += item['text']
            else:
                result_text = str(raw_content)

            print(f"👁️ [Qwen-VL 提取成功]: 预览 -> {result_text[:40]}...")
            return result_text.strip()
        return ""
    except Exception:
        return ""


# ==========================================
# 2. 升级版双流 RAG 系统 (混合检索架构 + DPO 日志)
# ==========================================
class DualStreamAgentSystem:
    def __init__(self, db_path="./agent_final_db"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.text_collection = self.chroma_client.get_or_create_collection(name="product_knowledge")
        self.image_collection = self.chroma_client.get_or_create_collection(name="product_visuals")

        print("正在加载 CLIP 视觉模型...")
        self.clip_model = SentenceTransformer('clip-ViT-B-32')

        self.qa_log_file = "qa_feedback_logs.jsonl"
        if not os.path.exists(self.qa_log_file):
            with open(self.qa_log_file, 'w', encoding='utf-8') as f:
                pass

    def _get_qwen_embedding(self, text):
        resp = dashscope.TextEmbedding.call(
            model=dashscope.TextEmbedding.Models.text_embedding_v3,
            input=text
        )
        if resp.status_code == 200:
            return resp.output['embeddings'][0]['embedding']
        raise Exception(f"Qwen Embedding failed: {resp.message}")

    def get_database_stats(self):
        print("\n📊 --- 当前知识库状态 ---")
        print(f"文本知识库 (QA问答用): 共 {self.text_collection.count()} 条商品数据")
        print(f"视觉特征库 (以图搜图用): 共 {self.image_collection.count()} 张图片特征")
        print("------------------------\n")

    # 【修改】：增加 overwrite 参数，默认 False (开启增量更新模式)
    def build_knowledge_base(self, df, overwrite=False):
        print(f"开始构建混合隔离架构的双路知识库 (覆盖更新模式: {overwrite})...")
        for index, row in df.iterrows():
            product_id = str(row['id'])

            # 【核心逻辑】：增量更新。如果不要求强制覆盖，且库里已经有这个ID了，直接跳过省钱省时间
            if not overwrite:
                existing = self.text_collection.get(ids=[f"txt_{product_id}"])
                if existing['ids']:
                    print(f"⏩ 商品 ID: {product_id} 已在库中，执行跳过。")
                    continue

            print(f"\n--- 正在处理商品 ID: {product_id} ---")

            # 视觉特征入库
            main_url = row.get('商品主图URL地址', '')
            if pd.notna(main_url) and str(main_url).strip().lower() != 'nan':
                main_img = download_image_with_auth(str(main_url))
                if main_img:
                    try:
                        img_embedding = self.clip_model.encode(main_img).tolist()
                        self.image_collection.upsert(
                            documents=[f"Image of product {product_id}"],
                            embeddings=[img_embedding],
                            metadatas=[{"product_id": product_id, "url": main_url, "name": row['商品完整名称']}],
                            ids=[f"img_{product_id}"]
                        )
                        print("✅ 主图视觉特征入库成功")
                    except Exception:
                        pass

            searchable_base_info = f"商品名称：{row['商品完整名称']}，型号：{row['商品型号']}"

            image_specs_text = "无附加图纸参数"
            detail_urls_raw = row.get('商详图片', '')
            if pd.notna(detail_urls_raw) and str(detail_urls_raw).strip().lower() != 'nan':
                detail_urls = str(detail_urls_raw).split('\n')
                stitched_image = stitch_images_vertically(detail_urls)
                if stitched_image:
                    image_specs_text = extract_specs_from_stitched_image(stitched_image)

            try:
                # 【修改标注 2】：混合检索策略 Hybrid Embedding
                # 将 Qwen-VL 提取的说明文字的前 150 个字符（摘要关键词）拼接到基础信息中参与 Embedding！
                # 这样 "60mm"、"急停" 这些关键维度就能在向量空间中被搜索到了。
                vector_search_text = searchable_base_info
                if image_specs_text and "无附加图纸" not in image_specs_text:
                    # 将精华摘要揉入 Embedding 文本
                    vector_search_text += f" 特征摘要：{image_specs_text[:150]}"

                text_embedding = self._get_qwen_embedding(vector_search_text)

                self.text_collection.upsert(
                    documents=[searchable_base_info],  # 给人类和大模型看的展示文本依然保持清爽
                    embeddings=[text_embedding],  # 但它的向量坐标已经包含了 60mm 等核心特征
                    metadatas=[{
                        "product_id": product_id,
                        "model": str(row['商品型号']),
                        "price": str(row['当前实际销售价格']),
                        "detailed_specs": image_specs_text  # 完整的上千字的详细参数依然藏在 Metadata 里
                    }],
                    ids=[f"txt_{product_id}"]
                )
                print("✅ 文本基础特征与参数 Metadata 入库成功")
            except Exception as e:
                print(f"❌ 入库失败: {e}")

    def search_by_image(self, user_image_path, top_k=3):
        try:
            user_img = Image.open(user_image_path).convert('RGB')
            query_embedding = self.clip_model.encode(user_img).tolist()
            results = self.image_collection.query(query_embeddings=[query_embedding], n_results=top_k)

            if not results['metadatas'] or not results['metadatas'][0]: return []

            matches = []
            for i in range(len(results['metadatas'][0])):
                matches.append({
                    'product_id': results['metadatas'][0][i]['product_id'],
                    'name': results['metadatas'][0][i]['name'],
                    'distance': results['distances'][0][i]
                })
            return matches
        except Exception:
            return []

    def chat_with_agent(self, user_query, chat_history=None):
        if chat_history is None: chat_history = []

        query_embedding = self._get_qwen_embedding(user_query)
        results = self.text_collection.query(query_embeddings=[query_embedding], n_results=3)

        context_parts = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                part = f"【基础信息】{doc}\n【价格】{meta.get('price', '未知')}元\n【图纸详情】{meta.get('detailed_specs', '无')}"
                context_parts.append(part)

        context = "\n---\n".join(context_parts) if context_parts else "知识库中暂未找到相关信息。"

        system_prompt = f"""
        你是一个专业的工业电子配件AI客服。请基于以下知识库上下文精准回答用户。

        【产品知识库上下文】：
        {context}

        【你的任务和规则】：
        1. 态度专业、礼貌。回答用户参数问题时，严格参考上下文。
        2. 如果上下文没提，直接说“很抱歉，当前资料中未说明该参数”。
        3. 【订单识别】如果用户明确表达下单意图，结合对话确认【型号】和【数量】。
        4. 如果确认用户下单，在正常回复末尾，附加 JSON 格式订单数据，用 ```json 和 ``` 包裹。
        示例：
        ```json
        {{"intent": "create_order", "model": "提取的型号", "quantity": 50, "unit_price": 100}}
        ```
        """

        messages = [{'role': 'system', 'content': system_prompt}]
        for turn in chat_history: messages.append(turn)
        messages.append({'role': 'user', 'content': user_query})

        response = dashscope.Generation.call(model='qwen-turbo', messages=messages, result_format='message')
        bot_reply = response.output.choices[0]['message']['content']

        chat_history.append({'role': 'user', 'content': user_query})
        chat_history.append({'role': 'assistant', 'content': bot_reply})

        order_json = None
        json_match = re.search(r'```json\n(.*?)\n```', bot_reply, re.DOTALL)
        if json_match:
            try:
                order_json = json.loads(json_match.group(1))
                bot_reply = re.sub(r'```json\n.*?\n```', '', bot_reply, flags=re.DOTALL).strip()
            except Exception:
                pass

        interaction_id = str(uuid.uuid4())
        log_entry = {
            "interaction_id": interaction_id,
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "user_query": user_query,
            "model_response": bot_reply,
            "human_corrected_response": "",
            "rating": None
        }
        with open(self.qa_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

        return bot_reply, order_json, chat_history, interaction_id

    def submit_feedback(self, interaction_id, rating, corrected_reply=""):
        """人工反馈接口：用于收集 DPO 微调数据"""
        logs = []
        with open(self.qa_log_file, 'r', encoding='utf-8') as f:
            for line in f: logs.append(json.loads(line))

        for log in logs:
            if log["interaction_id"] == interaction_id:
                log["rating"] = rating
                if corrected_reply: log["human_corrected_response"] = corrected_reply
                break

        with open(self.qa_log_file, 'w', encoding='utf-8') as f:
            for log in logs: f.write(json.dumps(log, ensure_ascii=False) + '\n')
        print(f"📝 感谢反馈！交互 [{interaction_id}] 的修正数据已存入微调库。")


# ==========================================
# 3. 运行 Demo 测试
# ==========================================
if __name__ == "__main__":
    df = pd.read_excel('../data/goods_batch_1_20260130_180359.xlsx').head(2)
    agent_system = DualStreamAgentSystem(db_path="./agent_final_db")
    agent_system.build_knowledge_base(df)
    agent_system.get_database_stats()

    # --- 场景 1：以图搜图 ---
    print("\n" + "=" * 50 + "\n🌟 测试场景 1：以图搜图 🌟\n" + "=" * 50)
    test_img_url = df.iloc[0]['商品主图URL地址']
    test_img = download_image_with_auth(test_img_url)
    if test_img:
        test_img_path = "mock_user_upload.jpg"
        test_img.save(test_img_path)
        matches = agent_system.search_by_image(test_img_path, top_k=3)
        if matches:
            for i, match in enumerate(matches):
                print(f"[{i + 1}] 【{match['name']}】 ID:{match['product_id']} | 距离:{match['distance']:.4f}")
        if os.path.exists(test_img_path): os.remove(test_img_path)

    # --- 场景 2：多轮对话与 RLHF 数据收集 ---
    print("\n" + "=" * 50 + "\n🛒 测试场景 2：多轮对话与订单、强化学习反馈 🛒\n" + "=" * 50)
    chat_history = []

    q1 = "我想找一款施耐德的开关，你们有 XD2PA24CR 吗？"
    print(f"👤 客户：{q1}")
    reply1, order1, chat_history, i_id1 = agent_system.chat_with_agent(q1, chat_history)
    print(f"🤖 客服：{reply1}\n")

    q2 = "这个操作位置是怎么样的？多少钱一个？"
    print(f"👤 客户：{q2}")
    reply2, order2, chat_history, i_id2 = agent_system.chat_with_agent(q2, chat_history)
    print(f"🤖 客服：{reply2}\n")

    # 模拟管理员后台介入 (强化学习数据收集)
    print(">>> [模拟管理员后台介入]：对上述回答话术不满意，提交人类修正版话术...")
    perfect_reply = "这款产品支持2个操作位置，当前参考售价是 150 元。这是热销款，今天下单明天发货，您看需要几个？"
    agent_system.submit_feedback(interaction_id=i_id2, rating=-1, corrected_reply=perfect_reply)

    q3 = "行吧，价格合适。那给我来 50 个吧，我急用。"
    print(f"\n👤 客户：{q3}")
    reply3, order3, chat_history, i_id3 = agent_system.chat_with_agent(q3, chat_history)
    print(f"🤖 客服：{reply3}\n")

    if order3:
        print("🎉 [后台系统日志]：成功拦截到大模型生成的结构化订单请求！")
        print(json.dumps(order3, indent=4, ensure_ascii=False))