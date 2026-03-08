import pandas as pd
import numpy as np
import cv2
import random
from PIL import Image, ImageDraw


# ==========================================
# 1. 图像处理工具箱 (Padding & ORB)
# ==========================================
class ImageUtils:
    def __init__(self):
        # 初始化 ORB 检测器 (特征点检测)
        self.orb = cv2.ORB_create(nfeatures=500)
        # 初始化 暴力匹配器 (Hamming距离适用于ORB)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def preprocess_image_padding(self, image_pil, target_size=(224, 224), fill_color=(255, 255, 255)):
        """
        Letterbox Resize: 保持比例缩放，空余部分填充白色。
        【核心作用】：防止长条形工业品（如导轨、长开关）被强制拉伸成正方形，导致特征丢失。
        """
        image = image_pil.convert("RGB")
        w, h = image.size

        # 计算缩放比例
        ratio = min(target_size[0] / w, target_size[1] / h)
        new_size = (int(w * ratio), int(h * ratio))

        # 缩放
        img_resized = image.resize(new_size, Image.LANCZOS)

        # 创建新画布并粘贴到中心
        new_img = Image.new("RGB", target_size, fill_color)
        paste_pos = ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2)
        new_img.paste(img_resized, paste_pos)

        return new_img

    def compute_orb_score(self, img1_pil, img2_pil):
        """
        计算两张图片的 ORB 特征匹配点数量 (用于精排)
        """
        try:
            # 转 OpenCV 格式 (RGB -> Gray)
            img1 = cv2.cvtColor(np.array(img1_pil), cv2.COLOR_RGB2GRAY)
            img2 = cv2.cvtColor(np.array(img2_pil), cv2.COLOR_RGB2GRAY)

            # 检测特征点
            kp1, des1 = self.orb.detectAndCompute(img1, None)
            kp2, des2 = self.orb.detectAndCompute(img2, None)

            if des1 is None or des2 is None:
                return 0

            # 匹配特征点
            matches = self.bf.match(des1, des2)

            # 返回匹配成功的点数
            return len(matches)
        except Exception as e:
            return 0


# 全局工具实例
img_utils = ImageUtils()


# ==========================================
# 2. 模拟组件 (Mock Clip Model & OCR)
# ==========================================
# 注意：在您的本地环境中，请替换为真实的 sentence_transformers 和 PaddleOCR

class MockClipModel:
    def encode(self, img):
        """
        模拟 CLIP 编码。
        为了让 Demo 跑通，这里返回随机向量。
        在真实环境中：return model.encode(img)
        """
        vec = np.random.rand(512)
        return vec / np.linalg.norm(vec)


class OCRService:
    def extract_text(self, image_obj):
        # 模拟 OCR 返回
        if random.random() > 0.7:
            return "额定电压:220V 绝缘等级:Class-B"
        return ""

    def clean_text(self, raw_text):
        return raw_text.strip().replace("\n", " ")


ocr_service = OCRService()


# ==========================================
# 3. 双流处理器 (读取 Excel 数据)
# ==========================================
class DualStreamProcessor:
    def __init__(self, clip_model):
        self.model = clip_model
        self.image_vector_db = []
        self.text_knowledge_buffer = []

    def download_image(self, url):
        """
        模拟下载图片。
        由于无法联网，这里生成一张带随机图案的模拟图。
        在真实环境中：return Image.open(requests.get(url, stream=True).raw)
        """
        try:
            # 生成一张随机图用于演示
            seed = sum([ord(c) for c in str(url)]) % 1000
            random.seed(seed)
            img = Image.new('RGB', (200, 200),
                            color=(random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)))
            draw = ImageDraw.Draw(img)
            # 画个框模拟产品
            draw.rectangle(
                [random.randint(10, 50), random.randint(10, 50), random.randint(100, 150), random.randint(100, 150)],
                fill=(0, 0, 0))
            return img
        except:
            return None

    def process_row(self, row):
        product_id = row['id']

        # 1. 处理主图
        main_url = row.get('商品主图URL地址', '')
        if pd.notna(main_url):
            img = self.download_image(main_url)
            if img:
                # 【关键步骤】入库前先 Padding
                padded_img = img_utils.preprocess_image_padding(img)
                vec = self.model.encode(padded_img)

                self.image_vector_db.append({
                    "product_id": product_id,
                    "vector": vec,
                    "img_type": "main",
                    "url": main_url,
                    # 真实场景中不要存 debug_img_obj 到内存，这里是为了演示搜索时能拿到原图
                    "debug_img_obj": img
                })

        # 2. 处理商详图片
        detail_urls_str = row.get('商详图片', '')
        if pd.notna(detail_urls_str):
            urls = str(detail_urls_str).split('\n')
            extracted_texts = []
            for url in urls:
                if not url.strip(): continue
                img = self.download_image(url)
                if not img: continue

                # 【关键步骤】商详图也做 Padding
                padded_img = img_utils.preprocess_image_padding(img)
                vec = self.model.encode(padded_img)

                self.image_vector_db.append({
                    "product_id": product_id,
                    "vector": vec,
                    "img_type": "detail",
                    "url": url,
                    "debug_img_obj": img
                })

                # OCR 提取
                raw_text = ocr_service.extract_text(img)
                if raw_text:
                    clean_txt = ocr_service.clean_text(raw_text)
                    if clean_txt: extracted_texts.append(clean_txt)

            if extracted_texts:
                combined_ocr_text = " ".join(extracted_texts)
                self.text_knowledge_buffer.append({
                    "product_id": product_id,
                    "ocr_content": combined_ocr_text
                })

    def get_databases(self):
        return self.image_vector_db, self.text_knowledge_buffer


# ==========================================
# 4. 搜索引擎 (含 ORB 重排)
# ==========================================
def visual_search_engine_optimized(user_query_img, vector_db, processor_instance, threshold=0.75, top_k=3, recall_n=20):
    # 【关键步骤】用户图也做 Padding
    query_img_padded = img_utils.preprocess_image_padding(user_query_img)
    query_vec = processor_instance.model.encode(query_img_padded)

    # 1. 向量召回 (CLIP Recall)
    candidates = []
    for item in vector_db:
        # 计算余弦相似度
        score = np.dot(query_vec, item['vector']) / (
                np.linalg.norm(query_vec) * np.linalg.norm(item['vector']) + 1e-9
        )
        candidates.append({**item, "clip_score": score})

    candidates.sort(key=lambda x: x['clip_score'], reverse=True)

    # 策略: 优先看主图
    main_subset = [c for c in candidates if c['img_type'] == 'main']

    if main_subset and main_subset[0]['clip_score'] > threshold:
        print(f"模式: 主图精搜 (Top 1 Score: {main_subset[0]['clip_score']:.4f})")
        recall_pool = main_subset[:recall_n]
    else:
        print(f"模式: 全局混搜 (Top 1 Score: {candidates[0]['clip_score']:.4f})")
        recall_pool = candidates[:recall_n]

    # 2. ORB 重排 (Re-rank)
    print(f"正在对 Top {len(recall_pool)} 个结果进行 ORB 视觉精排...")

    reranked_results = []
    for item in recall_pool:
        # 获取候选图原图 (Demo中直接从内存取，实际需下载)
        candidate_img = item.get('debug_img_obj')
        if not candidate_img:
            # 如果没存对象，就重新生成/下载
            candidate_img = processor_instance.download_image(item['url'])

        orb_score = 0
        if candidate_img:
            # 计算 用户原图 vs 候选原图 的特征点匹配数
            orb_score = img_utils.compute_orb_score(user_query_img, candidate_img)

        # 归一化 ORB 分数 (假设 50 个匹配点为满分)
        normalized_orb = min(orb_score / 50.0, 1.0)

        # 融合分数: 70% 语义相似 + 30% 纹理细节
        final_score = (item['clip_score'] * 0.7) + (normalized_orb * 0.3)

        reranked_results.append({
            **item,
            "final_score": final_score,
            "orb_matches": orb_score
        })

    reranked_results.sort(key=lambda x: x['final_score'], reverse=True)

    # 去重
    unique_results = []
    seen_ids = set()
    for res in reranked_results:
        if res['product_id'] not in seen_ids:
            unique_results.append(res)
            seen_ids.add(res['product_id'])
            if len(unique_results) >= top_k: break

    return unique_results


# =======================
# 运行 Demo
# =======================
if __name__ == "__main__":
    # 1. 读取 CSV (替换为您真实的文件路径)
    df = pd.read_excel('data/goods_batch_1_20260130_180359.xlsx')

    # 2. 初始化处理器
    processor = DualStreamProcessor(MockClipModel())

    print(f"正在读取并处理 Excel 数据 (共 {len(df)} 条)...")
    # 为了演示速度，这里只处理前 50 条
    for index, row in df.head(20).iterrows():
        processor.process_row(row)

    img_db, _ = processor.get_databases()
    print(f"数据库构建完成，共索引 {len(img_db)} 张图片。")

    # 3. 模拟用户搜索
    # 我们随机挑一张数据库里的图作为"用户上传图"
    if len(img_db) > 0:
        target_item = img_db[5]  # 假设用户要找这个商品
        print(f"\n--- 模拟用户搜索 ---")
        print(f"目标商品 ID: {target_item['product_id']}")

        # 模拟用户上传图：为了模拟真实拍摄，我们在原图上画个小黑框(噪声)
        user_query_img = target_item['debug_img_obj'].copy()
        draw = ImageDraw.Draw(user_query_img)
        draw.rectangle([0, 0, 20, 20], fill=(0, 0, 0))



        # 运行搜索
        results = visual_search_engine_optimized(
            user_query_img,
            img_db,
            processor_instance=processor,
            top_k=10
        )

        print("\n--- 最终搜索结果 ---")
        for i, res in enumerate(results):
            print(f"第 {i + 1} 名: ID={res['product_id']}")
            print(f"   [匹配来源]: {res['img_type']}")
            print(f"   [最终得分]: {res['final_score']:.4f}")
            print(f"   (CLIP语义分: {res['clip_score']:.4f}, ORB细节分: {res['orb_matches']}点)")
    else:
        print("未提取到图片数据，请检查 CSV 内容。")
