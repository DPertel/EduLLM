import argparse
import os
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from common import MODEL_ID, ROOT, read_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--train-file", default=str(ROOT / "data" / "train.jsonl"))
    parser.add_argument("--validation-file", default=str(ROOT / "data" / "validation.jsonl"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA não disponível. Treinamento cancelado para não usar CPU.")
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM total (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    torch.manual_seed(42)

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
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=max(args.rank * 2, 8),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def load_dataset(path):
        rows = read_jsonl(path)
        formatted = [
            {"text": f"### Instrução:\n{r['instruction']}\n\n### Resposta:\n{r['response']}{tokenizer.eos_token}"}
            for r in rows
        ]
        ds = Dataset.from_list(formatted)
        def tokenize(batch):
            return tokenizer(batch["text"], truncation=True, max_length=args.max_length)
        return ds.map(tokenize, batched=True, remove_columns=["text"])

    train_ds = load_dataset(args.train_file)
    eval_ds = load_dataset(args.validation_file)

    out_dir = ROOT / "outputs" / f"run_e{args.epochs}_lr{args.lr}_r{args.rank}_len{args.max_length}"
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        fp16=True,
        optim="paged_adamw_8bit",
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    result = trainer.train()
    trainer.save_model(str(ROOT / "artifacts" / "adapter"))
    tokenizer.save_pretrained(str(ROOT / "artifacts" / "adapter"))
    metrics = result.metrics
    metrics["gpu_name"] = torch.cuda.get_device_name(0)
    metrics["gpu_peak_memory_mb"] = round(torch.cuda.max_memory_allocated(0) / 1024**2, 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    print("Treinamento concluído. Adaptador salvo em artifacts/adapter")
    print(metrics)

if __name__ == "__main__":
    main()
