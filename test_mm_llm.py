from llama_index.multi_modal_llms.ollama import OllamaMultiModal
from llama_index.core.schema import ImageDocument

llm = OllamaMultiModal(model="llava", request_timeout=60.0)

prompt = "Analyze the image."
# We don't have an image ready, let's just see if complete works with text
resp = llm.stream_complete("Hello, are you there?", image_documents=[])
for r in resp:
    print(r.delta, end="", flush=True)

print("\nDone.")
