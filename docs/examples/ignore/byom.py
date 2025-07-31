from mellea import MelleaSession
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.base import LinearContext

m = MelleaSession(
    backend=OllamaModelBackend(model_id="llama3.2:latest"), ctx=LinearContext()
)

print(m.chat("Hi, meta! Tell me about Mark Zuckerberg."))
