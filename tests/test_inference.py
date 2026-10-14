"""Teste manual básico: execute depois de instalar dependências e treinar o adaptador."""
import torch

def test_cuda_required():
    assert torch.cuda.is_available(), "Este projeto exige GPU CUDA para os testes finais."
