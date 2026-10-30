import os
import subprocess
import sys
import time

from pyngrok import ngrok

# Coloque seu token aqui OU (recomendado) no secret NGROK_AUTHTOKEN do GitHub.
NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN") or "COLE_SEU_TOKEN_AQUI"

ngrok.set_auth_token(NGROK_AUTHTOKEN)
app = subprocess.Popen([sys.executable, "app.py"])
time.sleep(10)
tunnel = ngrok.connect(7860, "http")
print(f"\n=== URL PÚBLICA DO MAPA: {tunnel.public_url} ===\n", flush=True)

minutes = int(os.environ.get("RUN_MINUTES", "60"))
try:
    app.wait(timeout=minutes * 60)
except subprocess.TimeoutExpired:
    app.terminate()
