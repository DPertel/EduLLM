 <div align="center">

# 🤖 EduLLM — Tutor Local de Programação

### 🎓 Inteligência Artificial aplicada ao ensino de Python

Projeto acadêmico de fine-tuning local de uma LLM para tutoria de Python, com treinamento, avaliação experimental e interface interativa.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)

</div>

---

## 📌 Sobre o projeto

O **EduLLM** é um projeto acadêmico que explora o ajuste fino (*fine-tuning*) de um modelo de linguagem para atuar como tutor de programação em Python.

O projeto utiliza um modelo de linguagem local, técnicas de adaptação LoRA/QLoRA e uma interface web que permite enviar perguntas e analisar as respostas geradas.

### 🎯 Objetivos

- 🧠 Adaptar uma LLM para auxiliar no aprendizado de Python.
- 🖥️ Executar o modelo localmente utilizando uma GPU NVIDIA compatível.
- 🏋️ Experimentar técnicas de fine-tuning com LoRA/QLoRA.
- 📊 Avaliar a qualidade das respostas e o desempenho do modelo.
- 💬 Disponibilizar uma interface interativa para testar perguntas e parâmetros de geração.
- 🧪 Comparar diferentes configurações de treinamento e documentar os resultados.

## 🛠️ Tecnologias utilizadas

| Tecnologia | Finalidade |
|---|---|
| 🐍 Python | Linguagem principal do projeto |
| 🔥 PyTorch | Execução do modelo e operações na GPU |
| 🤗 Transformers | Carregamento e geração de texto |
| 🧩 PEFT / LoRA | Ajuste fino por adaptadores |
| ⚡ bitsandbytes | Quantização, conforme compatibilidade |
| 📚 Datasets | Preparação dos exemplos de treinamento |
| 🌐 Streamlit | Interface web interativa |

---

## 🖥️ Hardware-alvo e limitações

Configuração inicial para Windows 11, GPU NVIDIA com até 4 GB de VRAM e Python instalado.

- **Modelo-base:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Quantização:** 4 bits com bitsandbytes, quando compatível.
- **Método de adaptação:** LoRA/QLoRA.
- **Python recomendado:** 3.11 de 64 bits.

A viabilidade depende do modelo exato da GPU, do driver, do suporte CUDA, da compatibilidade das bibliotecas e da memória disponível. GPUs NVIDIA antigas podem não ser compatíveis com a versão atual do PyTorch ou do bitsandbytes.

> ⚠️ **Importante:** antes de começar o treinamento, execute `nvidia-smi` e `python src/gpu_check.py`. Se CUDA não estiver disponível, corrija o ambiente antes de prosseguir. Não use CPU para os testes que o professor exige que sejam feitos na GPU.

---

## 🚀 Como executar no Windows / VS Code

### 1️⃣ Preparar o ambiente

1. Instale ou atualize o driver NVIDIA apropriado à sua placa.
2. No terminal do VS Code, execute `nvidia-smi` e anote o modelo da GPU, a versão do driver e a memória disponível.
3. Utilize Python 3.10 ou 3.11 de 64 bits, recomendado para compatibilidade das bibliotecas.
4. Abra a pasta do projeto no VS Code e crie o ambiente virtual.

```powershell
py -3.11 -m venv .venv
```

Ative o ambiente virtual no PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Atualize o `pip`:

```powershell
python -m pip install --upgrade pip
```

Se o comando `py -3.11` não existir, instale Python 3.11 de 64 bits ou utilize uma versão disponível compatível.

#### 💡 Se a ativação da venv for bloqueada

Você pode executar diretamente o Python do ambiente virtual, sem alterar as políticas de execução do Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### 2️⃣ Instalar PyTorch com CUDA

Acesse o seletor oficial de instalação:

🔗 [PyTorch — Get Started](https://pytorch.org/get-started/locally/)

Selecione:

- **Operating System:** Windows
- **Package:** Pip
- **Language:** Python
- **Compute Platform:** versão CUDA compatível com seu driver e sua GPU

Execute o comando gerado pelo site.

> 💡 O PyTorch deve ser instalado especificamente para a GPU. Não instale `torch` novamente a partir de `requirements.txt`, pois a versão deve ser escolhida de acordo com a compatibilidade do ambiente.

### 3️⃣ Verificar se a GPU está funcionando

Execute:

```powershell
nvidia-smi
```

Em seguida, rode o teste do projeto:

```powershell
python src/gpu_check.py
```

O script deve mostrar:

- ✅ CUDA disponível: `True`
- 🎮 Nome da GPU detectada
- ⚙️ Dispositivo CUDA: `cuda:0`

Se o teste não confirmar a disponibilidade da CUDA, não prossiga para o treinamento.

### 4️⃣ Instalar as dependências

Com o ambiente virtual ativo e o PyTorch instalado, execute:

```powershell
python -m pip install -r requirements.txt
```

Se necessário, verifique as dependências:

```powershell
python -m pip check
```

O suporte a Windows do `bitsandbytes` depende da versão instalada e da compatibilidade da GPU. Se a instalação ou importação falhar, registre o erro e consulte as instruções oficiais:

🔗 [Documentação do bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)

Não substitua silenciosamente a execução em GPU por CPU.

---

## 📂 Estrutura do projeto

```text
EduLLM/
├── data/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── docs/
│   ├── metodologia.md
│   ├── resultados.md
│   └── screenshots/
├── frontend/
│   └── app.py
├── src/
│   ├── common.py
│   ├── gpu_check.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   └── test_inference.py
├── results/
├── requirements.txt
└── README.md
```

---

## 📚 4. Dataset

Os arquivos JSONL em `data/` são exemplos didáticos pequenos, utilizados inicialmente para validar o pipeline.

Para obter resultados acadêmicos mais robustos, amplie e revise o conjunto de dados. Evite duplicatas entre treino, validação e teste. O conjunto de teste não deve ser utilizado para ajustar o modelo.

### 📄 Formato dos dados

Cada linha deve conter um objeto JSON com os campos `instruction` e `response`.

Exemplo:

```json
{
  "instruction": "Explique o que é uma lista em Python.",
  "response": "Uma lista é uma coleção ordenada e mutável..."
}
```

---

## 🏋️ 5. Treinar com GPU

Após confirmar a disponibilidade da GPU e instalar as dependências, execute:

```powershell
python src/train.py --epochs 1 --lr 0.0001 --rank 4 --max-length 256
```

### ⚙️ Parâmetros utilizados

| Parâmetro | Valor inicial | Descrição |
|---|---:|---|
| `--epochs` | `1` | Quantidade de épocas |
| `--lr` | `0.0001` | Taxa de aprendizado |
| `--rank` | `4` | Rank dos adaptadores LoRA |
| `--max-length` | `256` | Comprimento máximo da sequência |

Esses parâmetros são uma configuração inicial conservadora para uma GPU de até 4 GB de VRAM. Mesmo assim, o treinamento pode exceder a memória disponível em algumas placas.

### ⚠️ Se ocorrer `CUDA out of memory`

1. 🛑 Pare a execução.
2. 📝 Registre a mensagem de erro.
3. 🔧 Reduza `--max-length` para `128`.
4. 🔄 Verifique a memória disponível e tente novamente, se apropriado.
5. ✅ Registre o treinamento como concluído somente se ele realmente terminar.

O modelo-base será baixado da Hugging Face na primeira execução. Depois de treinado, o adaptador LoRA será salvo em:

```text
artifacts/adapter/
```

---

## 📊 6. Avaliar o modelo

Execute:

```powershell
python src/evaluate.py --limit 20
```

A avaliação gera o arquivo:

```text
results/evaluation.csv
```

O CSV contém prompts, respostas de referência, respostas geradas pelo modelo e latência. A execução também confirma o dispositivo CUDA.


---

## 💬 7. Frontend — interface interativa

Depois de treinar o adaptador, inicie a interface:

```powershell
streamlit run frontend/app.py
```

O navegador abrirá a aplicação do EduLLM.

Na interface, você poderá:

1. ✍️ Digitar uma pergunta sobre Python.
2. 🎛️ Ajustar parâmetros como `temperature` e `top_p`.
3. 💬 Gerar e visualizar uma resposta.
4. 🧪 Experimentar diferentes configurações de geração.

O aplicativo carrega o modelo quantizado e o adaptador localmente. Ele não utiliza uma API externa para gerar respostas. Entretanto, o download inicial do modelo exige conexão com a internet.

---

## 🧪 8. Comparação experimental sugerida

Utilize o mesmo conjunto de perguntas e registre os resultados de cada execução.

| Execução | Épocas | Learning rate | LoRA rank | Max length |
|---|---:|---:|---:|---:|
| A | 1 | 0.0001 | 4 | 128 |
| B | 1 | 0.00005 | 4 | 128 |
| C | 2 | 0.0001 | 4 | 128 |
| D | 1 | 0.0001 | 8 | 128 |

Para cada execução, registre:

- ⏱️ Tempo total de treinamento
- 📉 Loss disponível
- 🎮 Memória de GPU utilizada
- ⚡ Latência das respostas
- 📝 Avaliação humana da qualidade

Compare somente execuções que realmente foram concluídas. Documente falhas e limitações e não invente resultados.

---

## 📸 9. Capturas de tela obrigatórias

O relatório de evidências salvo como:

`Relatorio_Treinando_LLM_Evidencias.pdf`

### 📋 Checklist das evidências

- [x] 🎮 **GPU:** terminal com `nvidia-smi`, modelo da placa e memória.
- [x] ✅ **CUDA:** execução do script `src/gpu_check.py`.
- [x] 🏋️ **Treinamento:** progresso ou saída final real do treinamento.
- [x] 💬 **Frontend:** interface aberta e resposta gerada para uma pergunta.
- [x] 📊 **Resultados:** avaliação executada e arquivo de resultados produzido.


---

## 🔐 Licença e reprodutibilidade

Verifique e documente a licença e os termos de uso do modelo-base e de todos os dados utilizados.

Para facilitar a reprodução dos experimentos, registre:

- 🖥️ Sistema operacional
- 🎮 Modelo da GPU e VRAM
- 🔧 Versão do driver
- 🐍 Versão do Python
- ⚡ Versões do PyTorch, CUDA e demais bibliotecas
- 🧪 Parâmetros utilizados em cada treinamento
- 🎲 Sementes aleatórias, quando aplicável
- 📚 Versão do dataset e critérios de avaliação

## 👥 Integrantes

- **Nome:** `Deivison Pertel` — **RM:** `550803`
- **Nome:** `Eduardo Akira Murata` — **RM:** `98713`
- **Nome:** `Wesley Souza de Oliveira` — **RM:** `97874`

**Curso:** Engenharia de Software  
**Disciplina:** GENERATIVE AI

---

<div align="center">

### 🚀 EduLLM — Aprendendo, testando e avaliando IA local

*Projeto acadêmico de Inteligência Artificial Generativa.*

</div>