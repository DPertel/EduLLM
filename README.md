# EduLLM — tutor local de programação

Projeto acadêmico de fine-tuning local de uma LLM para tutoria de Python.

## Hardware-alvo e limitações

Configuração inicial para Windows 11, GPU NVIDIA com até 4 GB de VRAM e Python instalado.
O modelo inicial é `Qwen/Qwen2.5-0.5B-Instruct`, carregado em 4 bits com bitsandbytes e adaptado por LoRA/QLoRA. A viabilidade depende do modelo exato da GPU, driver, suporte CUDA e memória disponível. GPUs NVIDIA antigas podem não ser compatíveis com a versão atual do PyTorch/bitsandbytes.

> Não comece o treinamento antes de executar `nvidia-smi` e `python src/gpu_check.py`. Se CUDA não estiver disponível, corrija o ambiente antes de prosseguir. Não use CPU para os testes que o professor exige que sejam feitos na GPU.

## 1. Preparar o Windows / VS Code

1. Instale/atualize o driver NVIDIA apropriado à sua placa.
2. No terminal do VS Code, execute `nvidia-smi` e anote modelo da GPU, versão do driver e memória.
3. Use Python 3.10 ou 3.11 x64 (recomendado para compatibilidade das bibliotecas).
4. Abra a pasta do projeto no VS Code e crie ambiente virtual:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Se o comando `py -3.11` não existir, instale Python 3.11 x64 ou use o executável Python disponível.

## 2. Instalar PyTorch com CUDA

Escolha a versão no seletor oficial: https://pytorch.org/get-started/locally/
Selecione Windows, Pip, Python e uma versão CUDA compatível com seu driver/GPU. Instale o comando que o site gerar. Não instale `torch` novamente a partir de `requirements.txt`, pois o PyTorch deve ser escolhido especificamente para a GPU.

Depois, valide:

```powershell
python src/gpu_check.py
```

O script deve mostrar `CUDA disponível: True`, o nome da GPU e `cuda:0`. Se não mostrar, não prossiga para treinamento.

## 3. Instalar dependências

```powershell
python -m pip install -r requirements.txt
```

O suporte a Windows do `bitsandbytes` depende da versão instalada e da compatibilidade da GPU. Se a instalação ou importação falhar, registre o erro; não substitua silenciosamente por CPU. Verifique as instruções oficiais do projeto: https://github.com/bitsandbytes-foundation/bitsandbytes

## 4. Dataset

Os arquivos JSONL em `data/` são exemplos didáticos pequenos, apenas para provar o pipeline. Para resultados acadêmicos, amplie e revise o conjunto de dados. Evite duplicatas entre treino, validação e teste. O conjunto de teste não deve ser usado para ajustar o modelo.

Formato: um objeto JSON por linha, com `instruction` e `response`.

## 5. Treinar com GPU

```powershell
python src/train.py --epochs 1 --lr 0.0001 --rank 4 --max-length 256
```

Parâmetros conservadores para uma GPU de até 4 GB: batch 1, gradient accumulation, sequência curta, rank 4 e checkpointing. Mesmo assim, o treinamento pode exceder a VRAM em algumas placas. Se ocorrer CUDA out of memory, pare, registre a falha e reduza `--max-length` para 128; não alegue que o treinamento concluiu.

O modelo-base é baixado da Hugging Face na primeira execução. Depois de treinado, o adaptador LoRA é salvo em `artifacts/adapter/`.

## 6. Avaliar

```powershell
python src/evaluate.py --limit 20
```

A avaliação gera `results/evaluation.csv` com prompt, resposta de referência, resposta do modelo e latência. Ela também confirma o dispositivo CUDA. Leia as respostas e avalie correção, clareza, aderência ao pedido e qualidade do exemplo. A métrica automática não substitui avaliação humana.

## 7. Frontend

Depois de treinar o adaptador:

```powershell
streamlit run frontend/app.py
```

Digite uma pergunta, selecione temperature/top-p e gere resposta. O app carrega o modelo quantizado e o adaptador localmente. Não usa API externa para gerar respostas.

## 8. Comparação experimental sugerida

Use o mesmo conjunto de perguntas e registre todos os resultados:

| Execução | Épocas | Learning rate | LoRA rank | Max length |
|---|---:|---:|---:|---:|
| A | 1 | 0.0001 | 4 | 128 |
| B | 1 | 0.00005 | 4 | 128 |
| C | 2 | 0.0001 | 4 | 128 |
| D | 1 | 0.0001 | 8 | 128 |

Só compare execuções que realmente concluíram. Registre tempo, loss, VRAM, latência e avaliação humana. Não invente resultados.

## 9. Capturas de tela obrigatórias

Salvo em `Relatorio_Treinando_LLM_Evidencias.pdf`:

## Licença e reprodutibilidade

Verifique e documente a licença e os termos do modelo-base e de todos os dados utilizados. Registre sistema, GPU, VRAM, driver, versões de bibliotecas, parâmetros e sementes em cada execução.
