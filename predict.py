import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Predictor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sesame/csm-1b")
        self.model = AutoModelForCausalLM.from_pretrained(
            "sesame/csm-1b",
            device_map="auto",
            torch_dtype=torch.float16,
        )

    def predict(self, prompt: str, max_length: int = 100) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
