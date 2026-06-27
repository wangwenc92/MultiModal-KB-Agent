from __future__ import annotations

"""微调模型推理服务"""
from pathlib import Path
from app.config import get_settings
from app.utils.logger import log

settings = get_settings()


class FinetunedService:
    """加载和使用微调后的模型"""

    def __init__(self, model_path: str = "data/finetuned_model"):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None

    def is_available(self) -> bool:
        return Path(self.model_path).exists() and any(Path(self.model_path).iterdir())

    def _load(self):
        if self._model is not None:
            return
        if not self.is_available():
            raise FileNotFoundError(f"Fine-tuned model not found at {self.model_path}")

        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        log.info(f"Loading fine-tuned model from {self.model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    def chat(self, question: str, system_prompt: str = "你是一个知识库助手。") -> str:
        self._load()
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        response = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response


_finetuned_service: FinetunedService | None = None


def get_finetuned_service() -> FinetunedService:
    global _finetuned_service
    if _finetuned_service is None:
        _finetuned_service = FinetunedService()
    return _finetuned_service
