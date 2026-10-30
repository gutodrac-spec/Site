"""Carrega e roda modelos do Hugging Face na GPU, otimizado para velocidade.

Tipos suportados:
  texto       -> LLMs de chat (transformers AutoModelForCausalLM)
  imagem      -> texto para imagem (diffusers)
  video       -> texto para vídeo (diffusers)
  fala        -> texto para fala (pipeline text-to-speech)
  transcrever -> fala para texto (pipeline automatic-speech-recognition)
  qualquer outra tarefa do transformers.pipeline (ex.: translation, summarization,
  image-classification, object-detection...)
"""
import base64
import gc
import io
import os
import threading
import uuid
from collections import OrderedDict

# Download paralelo e rápido dos pesos (hf_xet).
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

PASTA_SAIDA = os.environ.get("PASTA_SAIDA", "saidas")
MAX_MODELOS = int(os.environ.get("MAX_MODELOS", "2"))
COMPILAR = os.environ.get("COMPILAR", "0") == "1"

PADROES = {
    "texto": os.environ.get("MODELO_TEXTO", "Qwen/Qwen2.5-1.5B-Instruct"),
    "imagem": os.environ.get("MODELO_IMAGEM", "stabilityai/sdxl-turbo"),
    "video": os.environ.get("MODELO_VIDEO", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
    "fala": os.environ.get("MODELO_FALA", "facebook/mms-tts-por"),
    "transcrever": os.environ.get("MODELO_TRANSCRICAO", "openai/whisper-large-v3-turbo"),
}

TAREFAS_PIPELINE = {
    "fala": "text-to-speech",
    "transcrever": "automatic-speech-recognition",
}

_modelos = OrderedDict()  # repo_id -> {"tipo": ..., "obj": ...}
_trava = threading.Lock()  # uma geração por vez na GPU evita falta de memória
_configurado = False

os.makedirs(PASTA_SAIDA, exist_ok=True)


# ---------------------------------------------------------------- utilidades

def _torch():
    global _configurado
    import torch

    if not _configurado and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        _configurado = True
    return torch


def _dispositivo():
    return "cuda" if _torch().cuda.is_available() else "cpu"


def _dtype():
    torch = _torch()
    if not torch.cuda.is_available():
        return torch.float32
    # bf16 só é rápido a partir da arquitetura Ampere (A100, L4, RTX 30xx); T4 usa fp16.
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def _vram_gb():
    torch = _torch()
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def _liberar(repo_id):
    _modelos.pop(repo_id, None)
    gc.collect()
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def salvar_arquivo(dados: bytes, extensao: str) -> str:
    nome = f"{uuid.uuid4().hex}.{extensao}"
    with open(os.path.join(PASTA_SAIDA, nome), "wb") as f:
        f.write(dados)
    return nome


def _ler_audio(audio: str):
    """Aceita URL http(s), caminho de arquivo ou base64."""
    if audio.startswith(("http://", "https://")) or os.path.exists(audio):
        return audio
    if "," in audio and audio.startswith("data:"):
        audio = audio.split(",", 1)[1]
    return base64.b64decode(audio)


def _json_seguro(valor):
    """Converte saídas de pipelines (numpy, tensores, PIL) em algo serializável."""
    if isinstance(valor, dict):
        return {k: _json_seguro(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_json_seguro(v) for v in valor]
    if hasattr(valor, "save") and hasattr(valor, "size"):  # imagem PIL
        buf = io.BytesIO()
        valor.save(buf, format="PNG")
        return {"imagem_base64": base64.b64encode(buf.getvalue()).decode()}
    if hasattr(valor, "tolist"):
        return valor.tolist()
    return valor


# ---------------------------------------------------------------- carregamento

def _carregar_texto(repo_id, quantizar_4bit):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"torch_dtype": _dtype(), "device_map": "auto", "attn_implementation": "sdpa"}
    if quantizar_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=_dtype(), bnb_4bit_quant_type="nf4"
        )
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    modelo = AutoModelForCausalLM.from_pretrained(repo_id, **kwargs).eval()
    return {"tokenizer": tokenizer, "modelo": modelo}


def _carregar_imagem(repo_id):
    from diffusers import AutoPipelineForText2Image

    torch = _torch()
    pipe = AutoPipelineForText2Image.from_pretrained(repo_id, torch_dtype=_dtype())
    if _dispositivo() == "cuda" and _vram_gb() < 14 and "flux" in repo_id.lower():
        pipe.enable_model_cpu_offload()  # FLUX não cabe inteiro numa T4
    else:
        pipe.to(_dispositivo())
    pipe.set_progress_bar_config(disable=True)
    rede = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
    if rede is not None and _dispositivo() == "cuda":
        rede.to(memory_format=torch.channels_last)
        if COMPILAR:
            # Primeira geração demora mais; as seguintes ficam ~1,5-2x mais rápidas.
            setattr(pipe, "unet" if hasattr(pipe, "unet") else "transformer",
                    torch.compile(rede, mode="max-autotune-no-cudagraphs"))
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    # Aquecimento: a primeira geração real já sai na velocidade máxima.
    with torch.inference_mode():
        pipe(prompt="aquecimento", num_inference_steps=1, guidance_scale=0.0,
             width=512, height=512)
    return pipe


def _carregar_video(repo_id):
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=_dtype())
    if _dispositivo() == "cuda" and _vram_gb() < 20:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(_dispositivo())
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    return pipe


def _carregar_pipeline(repo_id, tarefa):
    from transformers import pipeline

    return pipeline(
        tarefa,
        model=repo_id,
        torch_dtype=_dtype(),
        device=0 if _dispositivo() == "cuda" else -1,
    )


def carregar(repo_id: str | None = None, tipo: str = "texto", quantizar_4bit: bool = False) -> str:
    repo_id = repo_id or PADROES.get(tipo)
    if not repo_id:
        raise ValueError(f"Informe o repo_id para o tipo '{tipo}'.")
    with _trava:
        if repo_id in _modelos:
            _modelos.move_to_end(repo_id)
            return f"{repo_id} já estava carregado ({tipo})."
        while len(_modelos) >= MAX_MODELOS:
            _liberar(next(iter(_modelos)))
        if tipo == "texto":
            obj = _carregar_texto(repo_id, quantizar_4bit)
        elif tipo == "imagem":
            obj = _carregar_imagem(repo_id)
        elif tipo == "video":
            obj = _carregar_video(repo_id)
        else:
            obj = _carregar_pipeline(repo_id, TAREFAS_PIPELINE.get(tipo, tipo))
        _modelos[repo_id] = {"tipo": tipo, "obj": obj}
    return f"{repo_id} carregado como '{tipo}' em {_dispositivo()}."


def descarregar(repo_id: str) -> str:
    with _trava:
        if repo_id not in _modelos:
            return f"{repo_id} não estava carregado."
        _liberar(repo_id)
    return f"{repo_id} removido da memória."


def _obter(tipo, repo_id=None):
    repo_id = repo_id or next((r for r, m in reversed(_modelos.items()) if m["tipo"] == tipo), None)
    if repo_id is None or repo_id not in _modelos:
        carregar(repo_id, tipo)
        repo_id = repo_id or PADROES[tipo]
    _modelos.move_to_end(repo_id)
    return _modelos[repo_id]["obj"]


# ---------------------------------------------------------------- geração

def conversar(mensagens: list[dict], max_tokens: int = 512, temperatura: float = 0.7,
              modelo: str | None = None) -> str:
    torch = _torch()
    m = _obter("texto", modelo)
    tok, llm = m["tokenizer"], m["modelo"]
    if tok.chat_template:
        texto = tok.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=True)
    else:
        texto = "\n".join(f"{x['role']}: {x['content']}" for x in mensagens) + "\nassistant:"
    entradas = tok(texto, return_tensors="pt").to(llm.device)
    opcoes = {"max_new_tokens": max_tokens, "pad_token_id": tok.pad_token_id or tok.eos_token_id}
    if temperatura > 0:
        opcoes.update(do_sample=True, temperature=temperatura, top_p=0.95)
    else:
        opcoes["do_sample"] = False
    with _trava, torch.inference_mode():
        saida = llm.generate(**entradas, **opcoes)
    return tok.decode(saida[0, entradas["input_ids"].shape[1]:], skip_special_tokens=True)


def gerar_texto(prompt: str, sistema: str = "", **kwargs) -> str:
    mensagens = ([{"role": "system", "content": sistema}] if sistema else [])
    mensagens.append({"role": "user", "content": prompt})
    return conversar(mensagens, **kwargs)


def gerar_imagem(prompt: str, passos: int = 2, guidance: float = 0.0, largura: int = 768,
                 altura: int = 768, seed: int = -1, negativo: str = "",
                 modelo: str | None = None) -> bytes:
    torch = _torch()
    pipe = _obter("imagem", modelo)
    opcoes = {"prompt": prompt, "num_inference_steps": passos, "guidance_scale": guidance,
              "width": largura, "height": altura}
    if negativo and guidance > 1:
        opcoes["negative_prompt"] = negativo
    if seed >= 0:
        opcoes["generator"] = torch.Generator(_dispositivo()).manual_seed(seed)
    with _trava, torch.inference_mode():
        img = pipe(**opcoes).images[0]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def gerar_video(prompt: str, quadros: int = 33, passos: int = 20, guidance: float = 5.0,
                largura: int = 832, altura: int = 480, fps: int = 16, seed: int = -1,
                modelo: str | None = None) -> str:
    """Gera um MP4 e devolve o nome do arquivo salvo em PASTA_SAIDA."""
    from diffusers.utils import export_to_video

    torch = _torch()
    pipe = _obter("video", modelo)
    opcoes = {"prompt": prompt, "num_frames": quadros, "num_inference_steps": passos,
              "guidance_scale": guidance, "width": largura, "height": altura}
    if seed >= 0:
        opcoes["generator"] = torch.Generator(_dispositivo()).manual_seed(seed)
    with _trava, torch.inference_mode():
        frames = pipe(**opcoes).frames[0]
    nome = f"{uuid.uuid4().hex}.mp4"
    export_to_video(frames, os.path.join(PASTA_SAIDA, nome), fps=fps)
    return nome


def falar(texto: str, modelo: str | None = None) -> bytes:
    """Texto para fala. Devolve um WAV."""
    import numpy as np
    from scipy.io import wavfile

    pipe = _obter("fala", modelo)
    with _trava, _torch().inference_mode():
        saida = pipe(texto)
    audio = np.asarray(saida["audio"]).squeeze()
    buf = io.BytesIO()
    wavfile.write(buf, saida["sampling_rate"], (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    return buf.getvalue()


def transcrever(audio: str, idioma: str | None = None, modelo: str | None = None) -> str:
    """Fala para texto. `audio` pode ser URL, caminho de arquivo ou base64."""
    pipe = _obter("transcrever", modelo)
    opcoes = {"chunk_length_s": 30, "batch_size": 8}
    if idioma:
        opcoes["generate_kwargs"] = {"language": idioma}
    with _trava, _torch().inference_mode():
        return pipe(_ler_audio(audio), **opcoes)["text"].strip()


def executar_pipeline(tarefa: str, repo_id: str, entrada, parametros: dict | None = None):
    """Roda qualquer tarefa do transformers.pipeline e devolve o resultado em JSON."""
    if repo_id not in _modelos:
        carregar(repo_id, tarefa)
    pipe = _modelos[repo_id]["obj"]
    with _trava, _torch().inference_mode():
        resultado = pipe(entrada, **(parametros or {}))
    return _json_seguro(resultado)


def status() -> dict:
    try:
        torch = _torch()
        if torch.cuda.is_available():
            livre, total = torch.cuda.mem_get_info()
            gpu = {"nome": torch.cuda.get_device_name(0),
                   "vram_livre_gb": round(livre / 1024**3, 1),
                   "vram_total_gb": round(total / 1024**3, 1)}
        else:
            gpu = {"nome": "sem GPU (CPU)"}
    except ImportError:
        gpu = {"nome": "torch não instalado"}
    return {"gpu": gpu,
            "modelos_carregados": {r: m["tipo"] for r, m in _modelos.items()},
            "padroes": PADROES,
            "max_modelos": MAX_MODELOS}
