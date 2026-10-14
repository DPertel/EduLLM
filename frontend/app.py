import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import streamlit as st
import torch
from common import load_model, generate_answer

st.set_page_config(page_title="EduLLM", page_icon="🎓", layout="centered")
st.title("🎓 EduLLM")
st.caption("Tutor local de programação Python — inferência em GPU NVIDIA")
if not torch.cuda.is_available():
    st.error("CUDA não está disponível. Verifique o driver e a instalação do PyTorch. Este app não usa CPU como fallback.")
    st.stop()

@st.cache_resource
def get_pipeline():
    return load_model(use_adapter=True)

try:
    tokenizer, model = get_pipeline()
except Exception as exc:
    st.error(f"Não foi possível carregar o modelo/adaptador: {exc}")
    st.info("Execute o treinamento primeiro: python src/train.py")
    st.stop()

st.success(f"GPU ativa: {torch.cuda.get_device_name(0)}")
prompt = st.text_area("Digite sua pergunta ou prompt", placeholder="Ex.: Explique listas em Python para um iniciante.", height=130)
col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.2, value=0.2, step=0.1)
with col2:
    top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
max_tokens = st.select_slider("Máximo de tokens novos", options=[64, 96, 128, 160, 192], value=128)

if st.button("Gerar resposta", type="primary", use_container_width=True):
    if not prompt.strip():
        st.warning("Digite uma pergunta antes de gerar.")
    else:
        with st.spinner("Gerando resposta na GPU..."):
            try:
                answer, seconds = generate_answer(tokenizer, model, prompt, temperature, top_p, max_tokens)
                st.subheader("Resposta")
                st.write(answer)
                st.caption(f"Latência: {seconds:.2f} s · Dispositivo: cuda:0")
                st.caption(f"VRAM alocada: {torch.cuda.memory_allocated(0) / 1024**2:.0f} MiB")
            except Exception as exc:
                st.error(f"Falha durante a inferência: {exc}")
