import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = ROOT / "artifacts" / "adapter"

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def load_model(use_adapter=True):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível. Este projeto exige inferência/testes em GPU.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=qconfig,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    if use_adapter:
        if not ADAPTER_DIR.exists():
            raise FileNotFoundError(
                f"Adaptador não encontrado em {ADAPTER_DIR}. Execute primeiro: python src/train.py"
            )
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model.eval()
    return tokenizer, model

def make_prompt(tokenizer, instruction):
    messages = [
        {"role": "system", "content": "Você é o EduLLM, um tutor de programação Python. Explique com clareza, use exemplos corretos e adapte a explicação a iniciantes. Se não souber, diga que não sabe."},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def generate_answer(tokenizer, model, instruction, temperature=0.2, top_p=0.9, max_new_tokens=160):
    prompt = make_prompt(tokenizer, instruction)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": float(temperature) > 0,
    }
    if float(temperature) > 0:
        kwargs["temperature"] = float(temperature)
        kwargs["top_p"] = float(top_p)
    torch.cuda.synchronize()
    import time
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), elapsed
