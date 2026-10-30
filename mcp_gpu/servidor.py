"""Expõe a GPU (Colab/Kaggle/RunPod/PC) como conector MCP do Claude e como API HTTP.

Tudo fica atrás de um token secreto que faz parte da URL:
  {base}/{TOKEN}/mcp                    -> conector personalizado do Claude
  {base}/{TOKEN}/api/...                -> API própria (texto, imagem, vídeo, fala, transcrição, pipeline)
  {base}/{TOKEN}/v1/...                 -> API compatível com OpenAI
  {base}/{TOKEN}/arquivos/{nome}        -> arquivos gerados (vídeos, áudios)
"""
import base64
import os
import secrets
import time
import uuid

from anyio import to_thread
from mcp.server.fastmcp import Audio, FastMCP, Image
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

import motor

TOKEN = os.environ.get("MCP_TOKEN") or secrets.token_urlsafe(24)
PORTA = int(os.environ.get("MCP_PORTA", "8000"))
BASE = f"/{TOKEN}"

mcp = FastMCP(
    "gpu-gratis",
    host="0.0.0.0",
    port=PORTA,
    streamable_http_path=f"{BASE}/mcp",
    stateless_http=True,
    # O ngrok muda o Host; a proteção fica por conta do token secreto na URL.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

estado = {"url_publica": f"http://localhost:{PORTA}"}


def _url_arquivo(nome: str) -> str:
    return f"{estado['url_publica']}{BASE}/arquivos/{nome}"


async def _rodar(funcao, *args, **kwargs):
    """Roda a geração numa thread para não travar o servidor."""
    return await to_thread.run_sync(lambda: funcao(*args, **kwargs))


# ======================================================================= MCP

@mcp.tool()
async def carregar_modelo(repo_id: str = "", tipo: str = "texto", quantizar_4bit: bool = False) -> str:
    """Baixa e carrega um modelo do Hugging Face na GPU.

    tipo: texto | imagem | video | fala | transcrever | qualquer tarefa do transformers.pipeline
    (ex.: translation, summarization, image-classification). repo_id vazio usa o modelo padrão do tipo.
    quantizar_4bit=True permite LLMs maiores (ex.: 7B-8B numa T4).
    """
    return await _rodar(motor.carregar, repo_id or None, tipo, quantizar_4bit)


@mcp.tool()
async def descarregar_modelo(repo_id: str) -> str:
    """Remove um modelo da memória da GPU."""
    return await _rodar(motor.descarregar, repo_id)


@mcp.tool()
async def gerar_texto(prompt: str, sistema: str = "", max_tokens: int = 512,
                      temperatura: float = 0.7, modelo: str = "") -> str:
    """Gera texto com um LLM aberto (padrão Qwen2.5-1.5B-Instruct)."""
    return await _rodar(motor.gerar_texto, prompt, sistema, max_tokens=max_tokens,
                        temperatura=temperatura, modelo=modelo or None)


@mcp.tool()
async def gerar_imagem(prompt: str, passos: int = 2, guidance: float = 0.0, largura: int = 768,
                       altura: int = 768, seed: int = -1, negativo: str = "",
                       modelo: str = "") -> Image:
    """Gera uma imagem (padrão SDXL-Turbo).

    Modelos turbo/schnell: 1-4 passos e guidance 0. Modelos normais: 20-30 passos e guidance 5-7.
    """
    dados = await _rodar(motor.gerar_imagem, prompt, passos, guidance, largura, altura, seed,
                         negativo, modelo or None)
    return Image(data=dados, format="jpeg")


@mcp.tool()
async def gerar_video(prompt: str, quadros: int = 33, passos: int = 20, guidance: float = 5.0,
                      largura: int = 832, altura: int = 480, seed: int = -1,
                      modelo: str = "") -> str:
    """Gera um vídeo curto (padrão Wan2.1 1.3B) e devolve o link do MP4. Pode levar alguns minutos."""
    nome = await _rodar(motor.gerar_video, prompt, quadros, passos, guidance, largura, altura,
                        seed=seed, modelo=modelo or None)
    return _url_arquivo(nome)


@mcp.tool()
async def gerar_audio(texto: str, modelo: str = "") -> Audio:
    """Transforma texto em fala (padrão MMS-TTS em português)."""
    return Audio(data=await _rodar(motor.falar, texto, modelo or None), format="wav")


@mcp.tool()
async def transcrever_audio(audio: str, idioma: str = "", modelo: str = "") -> str:
    """Transcreve áudio para texto (padrão Whisper large-v3-turbo). `audio`: URL ou base64."""
    return await _rodar(motor.transcrever, audio, idioma or None, modelo or None)


@mcp.tool()
async def executar_pipeline(tarefa: str, repo_id: str, entrada: str,
                            parametros: dict | None = None) -> dict:
    """Roda qualquer modelo do Hugging Face compatível com transformers.pipeline.

    Ex.: tarefa="translation", repo_id="Helsinki-NLP/opus-mt-en-pt", entrada="Hello world".
    """
    resultado = await _rodar(motor.executar_pipeline, tarefa, repo_id, entrada, parametros)
    return {"resultado": resultado}


@mcp.tool()
async def status() -> dict:
    """Mostra a GPU, a memória livre e os modelos carregados."""
    return motor.status()


# ================================================================ API própria

def _rota(caminho, metodos=("POST",)):
    return mcp.custom_route(f"{BASE}{caminho}", methods=list(metodos))


async def _corpo(request: Request) -> dict:
    try:
        return await request.json()
    except Exception:
        return {}


def _erro(e: Exception, codigo=500):
    return JSONResponse({"erro": str(e)}, status_code=codigo)


@_rota("/api/status", ("GET",))
async def api_status(request: Request):
    return JSONResponse(motor.status())


@_rota("/api/carregar")
async def api_carregar(request: Request):
    c = await _corpo(request)
    try:
        msg = await _rodar(motor.carregar, c.get("repo_id"), c.get("tipo", "texto"),
                           c.get("quantizar_4bit", False))
        return JSONResponse({"mensagem": msg})
    except Exception as e:
        return _erro(e)


@_rota("/api/descarregar")
async def api_descarregar(request: Request):
    c = await _corpo(request)
    return JSONResponse({"mensagem": await _rodar(motor.descarregar, c["repo_id"])})


@_rota("/api/texto")
async def api_texto(request: Request):
    c = await _corpo(request)
    try:
        if "mensagens" in c:
            texto = await _rodar(motor.conversar, c["mensagens"], c.get("max_tokens", 512),
                                 c.get("temperatura", 0.7), c.get("modelo"))
        else:
            texto = await _rodar(motor.gerar_texto, c["prompt"], c.get("sistema", ""),
                                 max_tokens=c.get("max_tokens", 512),
                                 temperatura=c.get("temperatura", 0.7), modelo=c.get("modelo"))
        return JSONResponse({"texto": texto})
    except Exception as e:
        return _erro(e)


@_rota("/api/imagem")
async def api_imagem(request: Request):
    """Devolve image/jpeg. Com "formato": "json" devolve base64."""
    c = await _corpo(request)
    try:
        dados = await _rodar(motor.gerar_imagem, c["prompt"], c.get("passos", 2),
                             c.get("guidance", 0.0), c.get("largura", 768), c.get("altura", 768),
                             c.get("seed", -1), c.get("negativo", ""), c.get("modelo"))
    except Exception as e:
        return _erro(e)
    if c.get("formato") == "json":
        return JSONResponse({"imagem_base64": base64.b64encode(dados).decode()})
    return Response(dados, media_type="image/jpeg")


@_rota("/api/video")
async def api_video(request: Request):
    c = await _corpo(request)
    try:
        nome = await _rodar(motor.gerar_video, c["prompt"], c.get("quadros", 33),
                            c.get("passos", 20), c.get("guidance", 5.0), c.get("largura", 832),
                            c.get("altura", 480), c.get("fps", 16), c.get("seed", -1),
                            c.get("modelo"))
        return JSONResponse({"url": _url_arquivo(nome)})
    except Exception as e:
        return _erro(e)


@_rota("/api/fala")
async def api_fala(request: Request):
    c = await _corpo(request)
    try:
        return Response(await _rodar(motor.falar, c["texto"], c.get("modelo")),
                        media_type="audio/wav")
    except Exception as e:
        return _erro(e)


@_rota("/api/transcrever")
async def api_transcrever(request: Request):
    """JSON {"audio": URL ou base64} ou multipart com campo "arquivo"."""
    try:
        if request.headers.get("content-type", "").startswith("multipart/"):
            form = await request.form()
            audio = base64.b64encode(await form["arquivo"].read()).decode()
            idioma, modelo = form.get("idioma"), form.get("modelo")
        else:
            c = await _corpo(request)
            audio, idioma, modelo = c["audio"], c.get("idioma"), c.get("modelo")
        return JSONResponse({"texto": await _rodar(motor.transcrever, audio, idioma, modelo)})
    except Exception as e:
        return _erro(e)


@_rota("/api/pipeline")
async def api_pipeline(request: Request):
    c = await _corpo(request)
    try:
        resultado = await _rodar(motor.executar_pipeline, c["tarefa"], c["repo_id"],
                                 c["entrada"], c.get("parametros"))
        return JSONResponse({"resultado": resultado})
    except Exception as e:
        return _erro(e)


@_rota("/arquivos/{nome}", ("GET",))
async def api_arquivo(request: Request):
    nome = os.path.basename(request.path_params["nome"])
    caminho = os.path.join(motor.PASTA_SAIDA, nome)
    if not os.path.isfile(caminho):
        return _erro(FileNotFoundError(nome), 404)
    return FileResponse(caminho)


# ================================================== API compatível com OpenAI
# Permite usar apps e bibliotecas feitas para OpenAI apontando base_url para {url}/{TOKEN}/v1

@_rota("/v1/models", ("GET",))
async def oa_modelos(request: Request):
    dados = [{"id": r, "object": "model", "owned_by": "local"} for r in motor.status()["modelos_carregados"]]
    return JSONResponse({"object": "list", "data": dados})


@_rota("/v1/chat/completions")
async def oa_chat(request: Request):
    c = await _corpo(request)
    modelo = c.get("model")
    if modelo in ("", "default", None) or "/" not in modelo:
        modelo = None
    mensagens = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str)
                  else " ".join(p.get("text", "") for p in m["content"])} for m in c["messages"]]
    try:
        texto = await _rodar(motor.conversar, mensagens, c.get("max_tokens") or 512,
                             c.get("temperature", 0.7), modelo)
    except Exception as e:
        return _erro(e)
    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": modelo or motor.PADROES["texto"],
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": texto}}],
    })


@_rota("/v1/images/generations")
async def oa_imagens(request: Request):
    c = await _corpo(request)
    largura, altura = (int(x) for x in c.get("size", "768x768").split("x"))
    modelo = c.get("model") if "/" in (c.get("model") or "") else None
    try:
        imagens = [await _rodar(motor.gerar_imagem, c["prompt"], 2, 0.0, largura, altura, -1, "",
                                modelo) for _ in range(c.get("n", 1))]
    except Exception as e:
        return _erro(e)
    return JSONResponse({"created": int(time.time()),
                         "data": [{"b64_json": base64.b64encode(i).decode()} for i in imagens]})


@_rota("/v1/audio/speech")
async def oa_fala(request: Request):
    c = await _corpo(request)
    modelo = c.get("model") if "/" in (c.get("model") or "") else None
    try:
        return Response(await _rodar(motor.falar, c["input"], modelo), media_type="audio/wav")
    except Exception as e:
        return _erro(e)


@_rota("/v1/audio/transcriptions")
async def oa_transcricao(request: Request):
    form = await request.form()
    modelo = form.get("model") if "/" in (form.get("model") or "") else None
    try:
        audio = base64.b64encode(await form["file"].read()).decode()
        texto = await _rodar(motor.transcrever, audio, form.get("language"), modelo)
    except Exception as e:
        return _erro(e)
    return JSONResponse({"text": texto})


# ================================================================== início

def iniciar(ngrok_token: str | None = None, pre_carregar: list[str] | None = None) -> dict:
    """Sobe o servidor em segundo plano, abre o túnel ngrok e devolve as URLs.

    pre_carregar: tipos para já deixar na GPU, ex.: ["texto", "imagem"].
    """
    import threading

    threading.Thread(target=lambda: mcp.run(transport="streamable-http"), daemon=True).start()
    if ngrok_token:
        from pyngrok import ngrok

        ngrok.set_auth_token(ngrok_token)
        estado["url_publica"] = ngrok.connect(PORTA, "http").public_url
    for tipo in pre_carregar or []:
        print(motor.carregar(None, tipo))
    base = f"{estado['url_publica']}{BASE}"
    urls = {"conector_claude": f"{base}/mcp", "api": f"{base}/api", "openai_base_url": f"{base}/v1"}
    print("\nNão compartilhe estas URLs, elas contêm o token de acesso:")
    for nome, url in urls.items():
        print(f"  {nome}: {url}")
    print()
    return urls


if __name__ == "__main__":
    iniciar(os.environ.get("NGROK_AUTHTOKEN"),
            [t for t in os.environ.get("PRE_CARREGAR", "").split(",") if t])
    while True:
        time.sleep(3600)
