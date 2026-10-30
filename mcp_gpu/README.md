# GPU no Claude + API

Roda modelos abertos do Hugging Face na sua própria GPU (Colab, Kaggle, RunPod ou seu PC) e expõe tudo como:
- **conector personalizado do Claude** (MCP);
- **API HTTP própria**;
- **API compatível com OpenAI** (funciona com o SDK `openai`, LangChain, Open WebUI, etc.).

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gutodrac-spec/Site/blob/main/mcp_gpu/gpu_no_claude.ipynb)

## Tipos de modelo e padrões
| Tipo | Modelo padrão | Observação |
|---|---|---|
| `texto` | `Qwen/Qwen2.5-1.5B-Instruct` | qualquer LLM de chat; `quantizar_4bit` para 7B-8B numa T4 |
| `imagem` | `stabilityai/sdxl-turbo` | qualquer modelo diffusers de texto→imagem |
| `video` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | leva alguns minutos numa T4 |
| `fala` | `facebook/mms-tts-por` | texto→fala |
| `transcrever` | `openai/whisper-large-v3-turbo` | fala→texto |
| qualquer tarefa do `transformers.pipeline` | — | `translation`, `summarization`, `image-classification`... |

## URLs
Ao iniciar, o servidor imprime (com `{base}` = URL do ngrok + token):
- `{base}/mcp` → conector do Claude
- `{base}/api` → API própria
- `{base}/v1` → API compatível com OpenAI

## API própria
| Rota | Corpo JSON | Resposta |
|---|---|---|
| `GET /api/status` | — | GPU, VRAM, modelos carregados |
| `POST /api/carregar` | `{"tipo": "texto", "repo_id": "...", "quantizar_4bit": false}` | mensagem |
| `POST /api/descarregar` | `{"repo_id": "..."}` | mensagem |
| `POST /api/texto` | `{"prompt": "...", "sistema": "", "max_tokens": 512, "temperatura": 0.7}` ou `{"mensagens": [...]}` | `{"texto": ...}` |
| `POST /api/imagem` | `{"prompt": "...", "passos": 2, "largura": 768, "altura": 768, "formato": "json"}` | JPEG (ou base64) |
| `POST /api/video` | `{"prompt": "...", "quadros": 33, "passos": 20}` | `{"url": ...mp4}` |
| `POST /api/fala` | `{"texto": "..."}` | WAV |
| `POST /api/transcrever` | `{"audio": "URL ou base64"}` ou multipart `arquivo` | `{"texto": ...}` |
| `POST /api/pipeline` | `{"tarefa": "translation", "repo_id": "Helsinki-NLP/opus-mt-en-pt", "entrada": "Hello"}` | `{"resultado": ...}` |

Todas aceitam `"modelo": "dono/nome"` para escolher outro modelo do Hugging Face.

## API compatível com OpenAI
```python
from openai import OpenAI
client = OpenAI(base_url="URL_openai_base_url", api_key="nao-usado")
r = client.chat.completions.create(model="default", messages=[{"role": "user", "content": "Olá!"}])
print(r.choices[0].message.content)
```
Rotas: `/v1/models`, `/v1/chat/completions`, `/v1/images/generations`, `/v1/audio/speech`, `/v1/audio/transcriptions`.
Em `model`, use `default` ou um repo do Hugging Face (`dono/nome`).

## Otimizações de velocidade
- Download paralelo dos pesos com **hf_xet**.
- **fp16** (T4) ou **bf16** (A100/L4/RTX 30+) automaticamente; TF32 e cuDNN benchmark ligados.
- Atenção **SDPA** nos LLMs; `channels_last` e **VAE tiling** nos modelos de imagem/vídeo.
- **Aquecimento** ao carregar: a primeira imagem já sai na velocidade máxima.
- Modelos ficam **na memória** entre chamadas (até `MAX_MODELOS`; o mais antigo sai quando precisa de espaço).
- Modelos padrão rápidos: SDXL-Turbo (1-4 passos), Whisper turbo, Qwen 1.5B.
- **Offload automático** para CPU quando o modelo não cabe na VRAM (FLUX/vídeo numa T4).
- `COMPILAR=1` ativa `torch.compile` nos modelos de imagem (primeira geração mais lenta, as seguintes ~1,5-2x mais rápidas).
- Gerações rodam em thread separada e uma por vez na GPU, evitando falta de memória.

## Variáveis de ambiente
`NGROK_AUTHTOKEN`, `MCP_TOKEN` (gerado se vazio), `MCP_PORTA` (8000), `PRE_CARREGAR` (ex.: `texto,imagem`), `MAX_MODELOS` (2), `COMPILAR` (0), `PASTA_SAIDA` (`saidas`), `MODELO_TEXTO`, `MODELO_IMAGEM`, `MODELO_VIDEO`, `MODELO_FALA`, `MODELO_TRANSCRICAO`.

## Rodar fora do Colab
```bash
pip install -r requirements.txt
NGROK_AUTHTOKEN=seu_token PRE_CARREGAR=texto,imagem python servidor.py
```

## Avisos
- As URLs contêm o token de acesso: quem tiver a URL usa sua GPU. Não compartilhe.
- Cada pessoa roda na própria conta. Respeite os termos de uso do Colab/Kaggle (o Colab grátis restringe uso como servidor remoto) e a licença de cada modelo (o SDXL-Turbo, por exemplo, restringe uso comercial).
