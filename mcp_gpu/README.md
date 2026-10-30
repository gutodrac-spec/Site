# GPU no Claude (conector MCP)

Roda um modelo de imagem na sua própria GPU (Colab, Kaggle, RunPod ou seu PC) e conecta no Claude como **conector personalizado**.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gutodrac-spec/Site/blob/main/mcp_gpu/gpu_no_claude.ipynb)

## Ferramentas
| Ferramenta | O que faz |
|---|---|
| `baixar_modelo(repo_id)` | Baixa e carrega um modelo de imagem do Hugging Face (padrão `stabilityai/sdxl-turbo`) |
| `gerar_imagem(prompt, passos, guidance, largura, altura, seed)` | Gera a imagem e devolve para o Claude |
| `status()` | Mostra a GPU e o modelo carregado |

## Uso
- **Colab/Kaggle:** abra `gpu_no_claude.ipynb`, ative a GPU, cole o token do ngrok e rode.
- **PC/servidor:**
  ```bash
  pip install -r requirements.txt
  NGROK_AUTHTOKEN=seu_token python servidor.py
  ```
Copie a URL impressa e adicione em Claude → Configurações → Conectores → Adicionar conector personalizado.

## Variáveis
- `NGROK_AUTHTOKEN` – token do ngrok
- `MCP_TOKEN` – token secreto da URL (gerado automaticamente se vazio)
- `MODELO_PADRAO` – modelo carregado por padrão
- `MCP_PORTA` – porta local (padrão 8000)

## Avisos
- A URL contém o token de acesso: quem tiver a URL usa sua GPU. Não compartilhe.
- Cada pessoa roda na própria conta. Respeite os termos de uso do Colab/Kaggle (o Colab grátis restringe uso como servidor remoto) e a licença de cada modelo.
