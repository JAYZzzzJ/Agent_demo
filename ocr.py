import cv2
import numpy as np
import logging
import sys
from paddleocr import PaddleOCR
from PIL import Image

# ==========================================
# 日志配置
# ==========================================
logging.getLogger("ppocr").setLevel(logging.WARNING)


class RealOCRService:
    def __init__(self, use_gpu=False):
        print("正在初始化 OCR 引擎 (这可能需要几秒钟)...")

        device_str = 'gpu' if use_gpu else 'cpu'

        try:
            # --- 关键修改点 1: 增加 det_limit_side_len ---
            # 默认是 960。对于商详图、长图、密集表格，必须调大！
            # 否则图片会被缩小，导致小字无法识别。建议设置为 2048 或更高。
            self.ocr_engine = PaddleOCR(
                use_textline_orientation=True,
                lang="ch",
                device=device_str,
                det_limit_side_len=2048  # <--- 这里改成了 2048，大幅提升长图/小字识别率
            )
            print(f"✅ OCR 引擎初始化完成 (设备: {device_str})")

        except Exception as e:
            print(f"❌ OCR 引擎初始化失败: {e}")
            raise e

    def extract_text(self, image_obj, min_confidence=0.5):  # <--- 关键修改点 2: 阈值降到 0.5
        """
        执行 OCR 识别
        :param min_confidence: 建议降低到 0.5 或 0.6，防止漏掉模糊的字
        """
        try:
            # --- 关键修改点 3: 强制转 RGB ---
            # 这一步能解决 PNG 透明图或灰度图导致的 numpy 转换错误
            image_obj = image_obj.convert('RGB')

            # 格式转换: PIL Image (RGB) -> Numpy Array (BGR)
            img_np = np.array(image_obj)
            img_np = img_np[:, :, ::-1]  # RGB 转 BGR

            # 调用 OCR
            result = self.ocr_engine.ocr(img_np, cls=True)

            # 调试打印：查看原始结果，方便排查
            # print(f"DEBUG: Raw Result: {result}")

            if not result or result[0] is None:
                print("⚠️ 警告: 未检测到任何文字区域")
                return ""

            valid_texts = []
            for line in result[0]:
                text_info = line[1]
                text_content = text_info[0]
                confidence = text_info[1]

                # 仅添加符合置信度的文本
                if confidence >= min_confidence:
                    valid_texts.append(text_content)
                # else:
                #     print(f"丢弃低置信度文本: {text_content} (置信度: {confidence:.2f})")

            return " ".join(valid_texts)

        except Exception as e:
            print(f"❌ OCR 识别出错: {e}")
            return ""

    def clean_text(self, raw_text):
        if not raw_text:
            return ""
        cleaned = raw_text.strip()
        blacklist = ["点击查看", "详情请咨询", "满减", "优惠券", "评价", "客服"]
        for word in blacklist:
            cleaned = cleaned.replace(word, "")
        return cleaned.strip()


# ==========================================
# 单元测试
# ==========================================
if __name__ == "__main__":
    import requests
    from io import BytesIO

    # 实例化
    ocr = RealOCRService(use_gpu=False)

    test_url = "https://img10.360buyimg.com/jdi/jfs/t1/243590/38/5546/34395/65e8344eF9db87643/fc4e999d69443760.jpg"

    print(f"正在下载测试图片: {test_url} ...")
    try:
        response = requests.get(test_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))

        print("正在识别...")
        # 注意：这里如果仍然识别为空，可以试着传入 min_confidence=0.1 来测试是否是阈值问题
        raw_text = ocr.extract_text(image, min_confidence=0.5)
        cleaned_text = ocr.clean_text(raw_text)

        print("\n" + "=" * 30)
        print("识别结果:")
        print("=" * 30)
        if not cleaned_text:
            print("(结果为空)")
        else:
            print(cleaned_text)
        print("=" * 30)

    except Exception as e:
        print(f"测试失败: {e}")