"""
LoRA微调脚本
支持对Qwen/LLaMA等开源模型进行轻量微调
"""
import sys
sys.path.insert(0, ".")

import json
import os
from app.utils.logger import log

# 微调配置
DEFAULT_CONFIG = {
    "base_model": "Qwen/Qwen2-7B-Instruct",
    "data_path": "data/training_data.jsonl",
    "output_dir": "data/finetuned_model",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4,
    "max_seq_length": 2048,
    "warmup_ratio": 0.1,
    "logging_steps": 10,
    "save_strategy": "epoch",
}


def load_training_data(data_path: str) -> list[dict]:
    """加载训练数据"""
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            data.append(record)
    log.info(f"Loaded {len(data)} training records from {data_path}")
    return data


def format_dataset(data: list[dict], tokenizer) -> list[dict]:
    """将训练数据格式化为模型输入"""
    formatted = []
    for record in data:
        messages = record["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted.append({"text": text})
    return formatted


def run_lora_finetune(config: dict = None):
    """执行LoRA微调"""
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer
        import torch
    except ImportError as e:
        log.error(f"Missing dependencies: {e}. Install: pip install transformers peft trl torch")
        return

    log.info(f"Starting LoRA fine-tuning with base model: {cfg['base_model']}")

    # 1. 加载tokenizer和模型
    log.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 2. 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 加载和格式化数据
    data = load_training_data(cfg["data_path"])
    dataset = format_dataset(data, tokenizer)

    # 4. 训练参数
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        fp16=True,
        gradient_accumulation_steps=4,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    # 5. 开始训练
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        tokenizer=tokenizer,
    )

    log.info("Starting training...")
    trainer.train()

    # 6. 保存模型
    os.makedirs(cfg["output_dir"], exist_ok=True)
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    log.info(f"Model saved to {cfg['output_dir']}")


def merge_lora_weights(base_model: str, lora_path: str, output_path: str):
    """合并LoRA权重到基础模型"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import torch
    except ImportError as e:
        log.error(f"Missing dependencies: {e}")
        return

    log.info(f"Merging LoRA weights: {lora_path} → {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, lora_path)

    model = model.merge_and_unload()
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    log.info(f"Merged model saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LoRA微调脚本")
    parser.add_argument("--base-model", type=str, default=DEFAULT_CONFIG["base_model"])
    parser.add_argument("--data", type=str, default=DEFAULT_CONFIG["data_path"])
    parser.add_argument("--output", type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_CONFIG["lora_rank"])
    parser.add_argument("--merge", type=str, default=None, help="合并LoRA权重，传入LoRA路径")
    args = parser.parse_args()

    if args.merge:
        merge_lora_weights(args.base_model, args.merge, args.output)
    else:
        config = {
            "base_model": args.base_model,
            "data_path": args.data,
            "output_dir": args.output,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "lora_rank": args.lora_rank,
        }
        run_lora_finetune(config)
