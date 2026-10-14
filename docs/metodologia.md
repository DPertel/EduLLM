# Metodologia

## Problema
Adaptar uma LLM pequena para atuar como tutor introdutório de programação Python, respondendo a perguntas com explicações, exemplos e exercícios.

## Abordagem
- Modelo-base: `Qwen/Qwen2.5-0.5B-Instruct`.
- Ajuste: LoRA sobre camadas de atenção e MLP.
- Quantização de base: 4-bit NF4 para reduzir uso de VRAM.
- Dataset: instruções e respostas revisadas pelo grupo, separadas em treino, validação e teste.
- Execução: CUDA; scripts interrompem se CUDA não estiver disponível.

## Variáveis experimentais
- épocas (`--epochs`)
- taxa de aprendizado (`--lr`)
- rank LoRA (`--rank`)
- comprimento máximo (`--max-length`)
- geração: temperature, top-p e max_new_tokens

## Métricas
- loss de treino e validação registrada pelo Trainer
- latência por prompt
- memória de GPU
- avaliação humana de correção, clareza, relevância e aderência à instrução (escala 1–5)

## Limitações
O dataset inicial é minúsculo e serve somente para validar o pipeline. Ele não é suficiente para demonstrar capacidade geral. O modelo pode alucinar, errar código ou responder de forma inconsistente. Resultados devem ser obtidos e relatados após a execução real.
