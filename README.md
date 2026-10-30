---
title: Mapa
emoji: 🗺️
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Mapa (Gradio)

- **Hugging Face Space:** roda `app.py` direto (CPU grátis, sem GPU).
- **GitHub Actions + ngrok:** Actions → "Mapa via ngrok" → Run workflow. A URL pública aparece no log.
  Token: crie o secret `NGROK_AUTHTOKEN` (Settings → Secrets → Actions) ou cole em `run_ngrok.py`.
