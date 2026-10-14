import sys
import torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())
print("CUDA usada para compilar PyTorch:", torch.version.cuda)

if not torch.cuda.is_available():
    print("ERRO: CUDA não está disponível. Não execute o treinamento exigido em GPU ainda.")
    sys.exit(1)

print("GPU:", torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print(f"VRAM total: {props.total_memory / 1024**3:.2f} GB")
print("Compute capability:", f"{props.major}.{props.minor}")

try:
    a = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    b = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    c = a @ b
    torch.cuda.synchronize()
    print("Teste de cálculo na GPU: OK")
    print("Dispositivo do resultado:", c.device)
    print(f"VRAM alocada pelo teste: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MiB")
except Exception as exc:
    print("Falha no teste GPU:", repr(exc))
    sys.exit(2)
