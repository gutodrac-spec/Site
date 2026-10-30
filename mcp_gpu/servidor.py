"""Servidor MCP que expõe a GPU do Colab/Kaggle/PC como conector personalizado do Claude.

Ferramentas: baixar_modelo, gerar_imagem, status.
O acesso é protegido por um token secreto que faz parte da URL.
"""
import io
import os
import secrets

from mcp.server.fastmcp import FastMCP, Image
from mcp.server.transport_security import TransportSecuritySettings

TOKEN = os.environ.get("MCP_TOKEN") or secrets.token_urlsafe(24)
PORTA = int(os.environ.get("MCP_PORTA", "8000"))
MODELO_PADRAO = os.environ.get("MODELO_PADRAO", "stabilityai/sdxl-turbo")

mcp = FastMCP(
    "gpu-gratis",
    host="0.0.0.0",
    port=PORTA,
    streamable_http_path=f"/{TOKEN}/mcp",
    stateless_http=True,
    # O ngrok muda o Host; a proteção fica por conta do token secreto na URL.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

estado = {"pipe": None, "modelo": None}


def _dispositivo():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@mcp.tool()
def baixar_modelo(repo_id: str = MODELO_PADRAO) -> str:
    """Baixa um modelo de imagem do Hugging Face (diffusers) e carrega na GPU.

    Exemplos: stabilityai/sdxl-turbo, black-forest-labs/FLUX.1-schnell.
    """
    import torch
    from diffusers import AutoPipelineForText2Image

    disp = _dispositivo()
    dtype = torch.float16 if disp == "cuda" else torch.float32
    estado["pipe"] = None
    if disp == "cuda":
        torch.cuda.empty_cache()
    pipe = AutoPipelineForText2Image.from_pretrained(repo_id, torch_dtype=dtype)
    estado["pipe"] = pipe.to(disp)
    estado["modelo"] = repo_id
    return f"Modelo {repo_id} carregado em {disp}."


@mcp.tool()
def gerar_imagem(
    prompt: str,
    passos: int = 2,
    guidance: float = 0.0,
    largura: int = 768,
    altura: int = 768,
    seed: int = -1,
) -> Image:
    """Gera uma imagem a partir do prompt com o modelo carregado.

    Modelos turbo/schnell: 1-4 passos e guidance 0. Modelos normais: 20-30 passos e guidance 5-7.
    """
    import torch

    if estado["pipe"] is None:
        baixar_modelo(MODELO_PADRAO)
    gerador = None
    if seed >= 0:
        gerador = torch.Generator(_dispositivo()).manual_seed(seed)
    img = estado["pipe"](
        prompt=prompt,
        num_inference_steps=passos,
        guidance_scale=guidance,
        width=largura,
        height=altura,
        generator=gerador,
    ).images[0]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return Image(data=buf.getvalue(), format="jpeg")


@mcp.tool()
def status() -> str:
    """Mostra a GPU disponível e o modelo carregado."""
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sem GPU (CPU)"
    except ImportError:
        gpu = "torch não instalado"
    return f"GPU: {gpu} | Modelo carregado: {estado['modelo'] or 'nenhum'}"


def iniciar(ngrok_token: str | None = None) -> str:
    """Sobe o servidor em segundo plano, abre o túnel ngrok e devolve a URL do conector."""
    import threading

    threading.Thread(target=lambda: mcp.run(transport="streamable-http"), daemon=True).start()
    if not ngrok_token:
        url = f"http://localhost:{PORTA}/{TOKEN}/mcp"
    else:
        from pyngrok import ngrok

        ngrok.set_auth_token(ngrok_token)
        url = f"{ngrok.connect(PORTA, 'http').public_url}/{TOKEN}/mcp"
    print("\nURL do conector (não compartilhe, ela contém o token):\n" + url + "\n")
    return url


if __name__ == "__main__":
    iniciar(os.environ.get("NGROK_AUTHTOKEN"))
    import time

    while True:
        time.sleep(3600)
