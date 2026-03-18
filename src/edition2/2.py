import base64
import os
import pandas as pd
import json
import chromadb
import pillow_avif
from PIL import Image
from io import BytesIO
import requests
import dashscope
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
import time
import re
from dotenv import load_dotenv  # 新增：用于加载本地环境变量

# ==========================================
# 0. 环境与 API 密钥配置
# ==========================================
# 【改进 1】：加载 .env 文件，防止 IDE 读不到环境变量
load_dotenv(override=True)

# 请注意保护你的明文 API Key，建议后续也放入环境变量中
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QWEN_API_KEY = os.environ.get("DASHSCOPE_API_KEY")

dashscope.api_key = QWEN_API_KEY

# 【修复 1】：注释或删除国际站节点配置！
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ==========================================
# 工具函数：带浏览器伪装的图片下载器
# ==========================================
def download_image_with_auth(url):
    """【修复 2】：添加 User-Agent 伪装，突破京东防盗链反爬机制"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://www.jd.com/'
    }
    try:
        resp = requests.get(url.strip(), headers=headers, timeout=10)
        # 必须确保 HTTP 状态码是 200（成功），否则说明被拦截了
        if resp.status_code == 200:
            return Image.open(BytesIO(resp.content)).convert('RGB')
        else:
            print(f"被反爬拦截: 状态码 {resp.status_code} -> {url}")
            return None
    except Exception as e:
        print(f"网络异常 {url}: {e}")
        return None

# ==========================================
# 1. 图像处理与视觉理解 (Gemini)
# ==========================================
def stitch_images_vertically(image_urls):
    images = []
    for url in image_urls:
        if not url.strip(): continue
        img = download_image_with_auth(url) # 使用新的下载函数
        if img:
            images.append(img)

    if not images:
        return None

    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights)

    stitched_img = Image.new('RGB', (max_width, total_height), color=(255, 255, 255))
    y_offset = 0
    for im in images:
        stitched_img.paste(im, (0, y_offset))
        y_offset += im.size[1]

    return stitched_img


def extract_specs_from_stitched_image(stitched_img):
    """
    使用 Qwen-VL (通义千问视觉大模型) 替代 Gemini 解析商详图片
    """
    if stitched_img is None: return {}

    # 1. 将拼接好的 PIL Image 转为 Base64 字符串
    buffered = BytesIO()
    stitched_img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{img_base64}"

    prompt = """
    你是一个资深的电气自动化与数据科学工程师。这是一张由多张切片拼接而成的完整产品长图。
    请仔细分析图中的技术图纸、尺寸标注和参数表格。
    提取其中所有有价值的技术参数（如电压、孔径、材质、接线方式等）。
    请务必只输出一个合法的 JSON 格式字典，键为参数名，值为参数内容。不要有任何多余的 Markdown 标记。
    """

    try:
        # 调用百炼平台的 Qwen-VL-Max 模型
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": image_url},
                    {"text": prompt}
                ]
            }
        ]

        response = dashscope.MultiModalConversation.call(
            model='qwen-vl-max',
            messages=messages,
            result_format='message'
        )

        if response.status_code == 200:
            raw_content = response.output.choices[0]['message']['content']

            # 【核心修复】：处理多模态模型返回的 List 结构
            result_text = ""
            if isinstance(raw_content, list):
                # 遍历 List 提取出文本部分
                for item in raw_content:
                    if 'text' in item:
                        result_text += item['text']
            else:
                result_text = str(raw_content)

            print(f"👁️ [Qwen-VL 提取结果]: {result_text[:50]}...")  # 打印前50个字预览
            return result_text.strip()

        else:
            print(f"❌ Qwen-VL 视觉解析失败: {response.message}")
            return ""

    except Exception as e:
        print(f"图像解析异常: {e}")
        return {}

# ==========================================
# 2. 双流 RAG 系统 (Qwen + CLIP)
# ==========================================
class DualStreamAgentSystem:
    def __init__(self, db_path="./ecommerce_dual_db"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.text_collection = self.chroma_client.get_or_create_collection(name="product_knowledge")
        self.image_collection = self.chroma_client.get_or_create_collection(name="product_visuals")

        print("正在加载 CLIP 视觉模型...")
        self.clip_model = SentenceTransformer('clip-ViT-B-32')

    def get_database_stats(self):
        """查看当前知识库的数据量"""
        text_count = self.text_collection.count()
        image_count = self.image_collection.count()
        print("\n📊 --- 当前知识库状态 ---")
        print(f"文本知识库 (QA问答用): 共 {text_count} 条商品数据")
        print(f"视觉特征库 (以图搜图用): 共 {image_count} 张图片特征")
        print("------------------------\n")
        return text_count, image_count

    def _get_qwen_embedding(self, text):
        resp = dashscope.TextEmbedding.call(
            model=dashscope.TextEmbedding.Models.text_embedding_v3,
            input=text
        )
        if resp.status_code == 200:
            return resp.output['embeddings'][0]['embedding']
        else:
            raise Exception(f"Qwen Embedding failed: {resp.message}")

    def build_knowledge_base(self, df):
        print("开始构建双路知识库...")
        for index, row in df.iterrows():
            product_id = str(row['id'])
            print(f"\n--- 正在处理商品 ID: {product_id} ---")

            # 去重：如果库里有了就不重复跑，省钱省时间
            # existing = self.text_collection.get(ids=[f"txt_{product_id}"])
            # if existing['ids']:
            #     print(f"商品 ID: {product_id} 已在知识库中，跳过。")
            #     continue

            print(f"\n--- 正在处理商品 ID: {product_id} ---")
            # 流向 A：视觉特征入库
            main_url = row.get('商品主图URL地址', '')
            if pd.notna(main_url) and str(main_url).strip().lower() != 'nan':
                main_img = download_image_with_auth(main_url) # 使用新的下载函数
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
                    except Exception as e:
                        print(f"❌ 主图向量化失败: {e}")

            # 流向 B：文本知识入库
  #############    基础知识问题待解决       ##############
            searchable_base_info = f"商品名称：{row['商品完整名称']}，型号：{row['商品型号']}，价格：{row['当前实际销售价格']}元。"
            detail_urls_raw = row.get('商详图片', '')
            # 过滤商详图片的 nan 空值
            if pd.isna(detail_urls_raw) or str(detail_urls_raw).strip().lower() == 'nan':
                detail_urls = []
            else:
                detail_urls = str(detail_urls_raw).split('\n')
            print(f"商详图片原信息：{detail_urls_raw}")
            print(f"商详图片信息：{detail_urls}")
            if detail_urls:
                stitched_image = stitch_images_vertically(detail_urls)

                if stitched_image:
                    print("✅ 商详切片拼接完成，发送至 LLM 进行参数解析...")
                    image_specs = extract_specs_from_stitched_image(stitched_image)
                else:
                    image_specs =  ""
            else:
                image_specs = ""


            try:
                # 只算基础信息的向量
                text_embedding = self._get_qwen_embedding(searchable_base_info)

                # 【修改标注 3】：改为 upsert 覆盖错误数据；把大模型提取的商详图片参数 image_specs 藏进 metadatas
                self.text_collection.upsert(
                    documents=[searchable_base_info],
                    embeddings=[text_embedding],
                    metadatas=[{
                        "product_id": product_id,
                        "model": str(row['商品型号']),
                        "price": str(row['当前实际销售价格']),
                        "detailed_specs": image_specs if image_specs else '无附加图纸参数'
                    }],
                    ids=[f"txt_{product_id}"]
                )
                print("✅ 文本基础特征与参数 Metadata 入库成功")
            except Exception as e:
                print(f"❌ 文本特征入库失败: {e}")

    def search_by_image(self, user_image_path, top_k=3):
        """用户输入一张图片路径，返回最相似的商品"""
        try:
            # 获取当前图片库的总数量
            total_image_count = self.image_collection.count()
            print(f"🔍 正在从知识库的 {total_image_count} 张商品主图中进行视觉比对...")
            print(f"正在分析用户上传的图片: {user_image_path} ...")
            user_img = Image.open(user_image_path).convert('RGB')
            query_embedding = self.clip_model.encode(user_img).tolist()

            results = self.image_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            if not results['metadatas'] or not results['metadatas'][0]:
                return []

            matches = []
            # 遍历返回的多个结果
            for i in range(len(results['metadatas'][0])):
                matches.append({
                    'product_id': results['metadatas'][0][i]['product_id'],
                    'name': results['metadatas'][0][i]['name'],
                    'distance': results['distances'][0][i]  # 距离越小越相似
                })
            return matches
        except Exception as e:
            print(f"以图搜图失败: {e}")
            return []

    def chat_with_agent(self, user_query, chat_history=None):
        """
                引入多轮对话记忆（chat_history），并具备订单意图识别能力
                """
        if chat_history is None:
            chat_history = []

        # 1. 向量检索，寻找产品知识
        query_embedding = self._get_qwen_embedding(user_query)
        results = self.text_collection.query(query_embeddings=[query_embedding], n_results=3)

        context_parts = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]  # 检索命中的基础信息
                meta = results['metadatas'][0][i]  # 绑定的 Metadata 数据

                part = f"【基础信息】{doc}\n【参考价格】{meta.get('price', '未知')}元\n【图纸与详细参数】\n{meta.get('detailed_specs', '无附加图纸参数')}"
                context_parts.append(part)

        context = "\n---\n".join(context_parts) if context_parts else "知识库中暂未找到相关信息。"

        # 2. 构造强大的 System Prompt (赋予其导购和结账能力)
        system_prompt = f"""
                你是一个专业的工业电子配件AI客服。请基于以下知识库上下文精准回答用户。

                【产品知识库上下文】：
                {context}

                【你的任务和规则】：
                1. 态度专业、礼貌。如果用户问产品参数，请参考上下文如实回答。
                2. 如果上下文中没有用户问的参数，请直接说“很抱歉，当前资料中未说明该参数”。
                3. 【订单识别】如果用户明确表达了“购买”、“下单”、“买x个”等意图，请结合之前的对话，确认他需要的【商品型号】和【数量】。
                4. 如果确认用户要下单，请在你的正常回复内容的**最末尾**，附加一段 JSON 格式的订单数据，必须用 ```json 和 ``` 包裹。
                JSON 格式如下：
                ```json
                {{
                    "intent": "create_order",
                    "model": "提取的产品型号",
                    "quantity": 提取的数量(数字),
                    "unit_price": 提取的单价(数字)
                }}
                ```
                """

        # 3. 组装历史消息上下文发给大模型
        messages = [{'role': 'system', 'content': system_prompt}]
        for turn in chat_history:
            messages.append(turn)

        messages.append({'role': 'user', 'content': user_query})

        # 4. 调用 Qwen 大模型
        response = dashscope.Generation.call(
            model='qwen-turbo',
            messages=messages,
            result_format='message'
        )

        bot_reply = response.output.choices[0]['message']['content']

        # 5. 更新对话历史
        chat_history.append({'role': 'user', 'content': user_query})
        chat_history.append({'role': 'assistant', 'content': bot_reply})

        # 6. 后处理：尝试从大模型回复中提取订单 JSON (如果存在)
        order_json = None
        json_match = re.search(r'```json\n(.*?)\n```', bot_reply, re.DOTALL)
        if json_match:
            try:
                order_json = json.loads(json_match.group(1))
                # 提取后，把 JSON 部分从展示给用户的文字中删掉，让回复更自然
                bot_reply = re.sub(r'```json\n.*?\n```', '', bot_reply, flags=re.DOTALL).strip()
            except Exception as e:
                pass

        return bot_reply, order_json, chat_history

# ==========================================
# 3. 运行 Demo 测试
# ==========================================
if __name__ == "__main__":
    df = pd.read_excel('data/goods_batch_1_20260130_180359.xlsx').head(2)

    agent_system = DualStreamAgentSystem(db_path="../../agent_final_db")
    agent_system.build_knowledge_base(df)

    print("\n============== 测试场景 2：基于长图解析的文字问答 ==============")
    question = "请问 XD2PA24CR 这个型号的开关信息是什么？"
    print(f"客户：{question}")
    answer = agent_system.chat_with_agent(question)
    print(f"\n客服：\n{answer}")

    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("🌟 测试功能 1：以图搜图 🌟")
    print("=" * 50)

    # 模拟：我们下载 CSV 里第一条商品的主图到本地，充当用户拍照上传的图片
    test_img_url = df.iloc[0]['商品主图URL地址']
    print(f"准备测试图... (源自: {test_img_url})")
    test_img = download_image_with_auth(test_img_url)
    if test_img:
        test_img_path = "mock_user_upload.jpg"
        test_img.save(test_img_path)

        # 执行搜图
        # 请求 Top 3 相似商品
        matches = agent_system.search_by_image(test_img_path, top_k=3)
        if matches:
            print("🎯 搜图完成！为您匹配到以下高度相似的商品：\n")
            for i, match in enumerate(matches):
                # 距离越小越好，如果是同一个商品，distance 通常为 0 或者极其接近 0
                similarity_score = 1 - match['distance']  # 简单的将距离反转为相似度(仅供展示参考)
                print(f"[{i + 1}] 商品名称：【{match['name']}】")
                print(f"    商品 ID：{match['product_id']} | 向量距离：{match['distance']:.4f}")
        else:
            print("未搜索到结果。")

        # 测完删掉临时图
        if os.path.exists(test_img_path): os.remove(test_img_path)

    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("🛒 测试功能 2：多轮对话与生成订单 🛒")
    print("=" * 50)

    chat_history = []  # 记录多轮对话上下文

    # 第一轮：询问参数（意图：咨询）
    q1 = "我想找一款施耐德的开关，你们有 XD2PA24CR 吗？它的操作位置是怎么样的？"
    print(f"👤 客户：{q1}")
    reply1, order1, chat_history = agent_system.chat_with_agent(q1, chat_history)
    print(f"🤖 客服：{reply1}\n")

    # 第二轮：询问价格（意图：进一步确认）
    q2 = "好的，这个多少钱一个？"
    print(f"👤 客户：{q2}")
    reply2, order2, chat_history = agent_system.chat_with_agent(q2, chat_history)
    print(f"🤖 客服：{reply2}\n")

    # 第三轮：决定购买（意图：触发下单机制）
    q3 = "行吧，价格合适。那给我来 50 个吧，我急用。"
    print(f"👤 客户：{q3}")
    reply3, order3, chat_history = agent_system.chat_with_agent(q3, chat_history)
    print(f"🤖 客服：{reply3}\n")

    # 检查后台是否拦截到了订单数据
    if order3:
        print("🎉 [后台系统日志]：成功拦截到大模型生成的结构化订单请求！")
        print(json.dumps(order3, indent=4, ensure_ascii=False))
        print(
            f"➡️ 系统下一步动作：自动跳转至支付结算页，总价: {order3.get('quantity', 0) * order3.get('unit_price', 0)} 元")