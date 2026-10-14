import argparse
import csv
from pathlib import Path
import torch
from common import ROOT, read_jsonl, load_model, generate_answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--base-only", action="store_true", help="Avalia o modelo-base sem adaptador LoRA")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA não disponível; avaliação cancelada.")
    test_path = ROOT / "data" / "test.jsonl"
    rows = read_jsonl(test_path)[:args.limit]
    tokenizer, model = load_model(use_adapter=not args.base_only)
    results = []
    for i, row in enumerate(rows, 1):
        answer, latency = generate_answer(tokenizer, model, row["instruction"], temperature=0, max_new_tokens=160)
        results.append({
            "id": i, "instruction": row["instruction"], "reference": row["response"],
            "prediction": answer, "latency_seconds": round(latency, 3),
            "gpu": torch.cuda.get_device_name(0), "adapter": not args.base_only,
        })
        print(f"[{i}/{len(rows)}] {row['instruction']}\n{answer}\n")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    filename = "base_evaluation.csv" if args.base_only else "evaluation.csv"
    with open(out / filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys() if results else [])
        if results:
            writer.writeheader()
            writer.writerows(results)
    print(f"Salvo: {out / filename}")
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM pico (MiB):", round(torch.cuda.max_memory_allocated(0) / 1024**2, 1))

if __name__ == "__main__":
    main()
