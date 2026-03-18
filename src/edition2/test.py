import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
QWEN_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "").strip()

try:
    print("\n正在使用百炼平台【国际站/新加坡节点】兼容模式测试...")

    # 关键修改：base_url 必须指向 intl 国际站域名
    client = OpenAI(
        api_key=QWEN_API_KEY,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{'role': 'user', 'content': '你好，请回复数字 1'}],
    )

    print("✅ API Key 测试成功！连接正常。")
    print(f"AI 回复内容: {completion.choices[0].message.content}")

except Exception as e:
    print(f"❌ 发生异常：{e}")