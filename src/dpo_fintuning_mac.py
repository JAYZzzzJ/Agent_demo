import json
import torch
import os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig


# 1. 加载 DPO 数据集
def load_dpo_dataset(log_file):
    data_dict = {"prompt": [], "chosen": [], "rejected": []}

    if not os.path.exists(log_file):
        print(f"❌ 找不到日志文件 {log_file}，请先在客服系统中多交互几次并产生人工修正数据。")
        return None

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            log = json.loads(line)
            # 提取被人类修正过的负面评价数据
            if log.get("rating") == -1 and log.get("human_corrected_response"):
                prompt = f"基于以下知识库上下文：\n{log['context']}\n用户问：{log['user_query']}\n你的回答是："
                chosen = log["human_corrected_response"]
                rejected = log["model_response"]

                data_dict["prompt"].append(prompt)
                data_dict["chosen"].append(chosen)
                data_dict["rejected"].append(rejected)

    print(f"✅ 成功提取出 {len(data_dict['prompt'])} 条偏好修正数据。")
    return Dataset.from_dict(data_dict)


# 2. 核心训练逻辑 (Mac Air 特化版)
def train_model_on_mac():
    # 使用 1.5B 的小参数模型，完美适配 Mac Air 内存
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    print("⏳ 正在加载模型与分词器 (首次运行需要下载，请保持网络通畅)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 指定 device_map="mps"，调用 Mac 的 M 芯片硬件加速
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Mac 对 float16 支持较好
        device_map="mps"
    )

    # 配置 LoRA (只训练极小部分参数)
    peft_config = LoraConfig(
        r=8,  # 调小秩，进一步节省内存
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # 只微调核心注意力层
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    dataset = load_dpo_dataset("qa_feedback_logs.jsonl")
    if dataset is None or len(dataset) == 0:
        print("停止训练：没有有效数据。")
        return

    # 极端优化训练参数，防止 Mac 崩溃
    # 已移除废弃的 use_mps_device 参数，新版 transformers 会自动接管 MPS
    training_args = DPOConfig(
        output_dir="./qwen-mac-finetuned",
        per_device_train_batch_size=1,  # 批次设为 1，极致省内存
        gradient_accumulation_steps=4,  # 靠累加梯度来弥补小批次
        learning_rate=5e-5,
        num_train_epochs=3,
        logging_steps=2,
        beta=0.1,
        remove_unused_columns=False
    )

    print("🚀 开始在 Apple Silicon 上进行 DPO 微调...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,  # 【修复】：适配新版 trl 库，取代原来的 tokenizer=tokenizer
    )

    # 启动炼丹！
    trainer.train()

    print("💾 训练完成！正在保存 Mac 上的第一版微调模型权重...")
    trainer.save_model("./qwen-mac-finetuned-final")
    print("🎉 成功生成！路径: ./qwen-mac-finetuned-final")


if __name__ == "__main__":
    train_model_on_mac()