import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig


# ------------------------------------------------------------------
# 1. 准备训练数据：从 QA 日志中提取偏好对 (Prompt, Chosen, Rejected)
# ------------------------------------------------------------------
def load_dpo_dataset(log_file):
    """
    将我们在 agent_demo.py 中收集的日志，转化为 DPO 偏好对
    """
    data_dict = {"prompt": [], "chosen": [], "rejected": []}

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            log = json.loads(line)
            # 只有人类判定为负面(rating=-1)且给出了修正答案的日志，才具有 DPO 微调价值
            if log.get("rating") == -1 and log.get("human_corrected_response"):
                # 构造 Prompt：还原大模型当时看到的系统提示和用户问题
                prompt = f"基于以下知识库上下文：\n{log['context']}\n用户问：{log['user_query']}\n你的回答是："

                # Chosen (好回答)：人类专家修正后的话术
                chosen = log["human_corrected_response"]

                # Rejected (坏回答)：大模型当时的原话
                rejected = log["model_response"]

                data_dict["prompt"].append(prompt)
                data_dict["chosen"].append(chosen)
                data_dict["rejected"].append(rejected)

    print(f"✅ 成功从日志中提取出 {len(data_dict['prompt'])} 条偏好修正数据用于微调。")
    return Dataset.from_dict(data_dict)


# ------------------------------------------------------------------
# 2. 模型训练管道：LoRA 微调 Qwen2.5-7B-Instruct
# ------------------------------------------------------------------
def train_model():
    # 选择开源底座模型 (下载自 HuggingFace 或 ModelScope)
    # 此处以最强的开源 7B 模型 Qwen2.5 为例
    model_id = "Qwen/Qwen2.5-7B-Instruct"

    print("⏳ 正在加载开源大模型与分词器 (请确保有 16GB 以上的 GPU 显存)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # DPO 需要一个特殊标识符填充
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 以 bfloat16 精度加载模型，节省显存
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # 【核心】：使用 LoRA 冻结大部分参数，只训练极小部分附加参数。
    # 这样一张普通的 RTX 4090 显卡就能炼出企业级大模型
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Qwen 注意力层
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 包装模型为 LoRA 结构
    model = get_peft_model(model, peft_config)

    # 准备数据集
    dataset = load_dpo_dataset("qa_feedback_logs.jsonl")
    if len(dataset) < 10:
        print("⚠️ 警告：有效偏好数据少于 10 条，为了效果建议收集更多人工修正数据后再训练。")

    # 拆分训练集和验证集 (简单起见，这里假设数据够多)
    # dataset = dataset.train_test_split(test_size=0.1)

    # 设定 DPO 训练参数
    training_args = DPOConfig(
        output_dir="./qwen-ecommerce-finetuned",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        logging_steps=10,
        beta=0.1,  # DPO 偏好惩罚系数 (越大概率差异越敏感)
        fp16=True,  # 使用混合精度加速
        remove_unused_columns=False
    )

    print("🚀 开始 DPO 偏好对齐微调...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=0.1,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # 启动训练
    trainer.train()

    # ------------------------------------------------------------------
    # 3. 导出交付物：保存最终微调好的模型权重
    # ------------------------------------------------------------------
    print("💾 训练完成！正在保存最终的私有化模型权重...")
    trainer.save_model("./qwen-ecommerce-finetuned-final")
    print("🎉 交付文件生成成功！路径: ./qwen-ecommerce-finetuned-final")
    print("下一步：您可以将这个文件夹打包，作为私有化大模型交付给客户，客户可以使用 vLLM 等框架直接本地离线运行它。")


if __name__ == "__main__":
    train_model()