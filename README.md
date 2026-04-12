diff --git a/README.md b/README.md
new file mode 100644
index 0000000000000000000000000000000000000000..6318bff05351dd83ec47ab2ef1c789db32f9c1d9
--- /dev/null
+++ b/README.md
@@ -0,0 +1,103 @@
+# Ollama RAG Chatbot + Evaluation
+
+This project gives you a local chatbot with retrieval (RAG) using:
+- **Ollama** for generation (`llama3.1`) and embeddings (`nomic-embed-text`)
+- **ChromaDB** for persistent vector storage
+- A small **evaluation** command to test retrieval hit-rate
+
+## Quick start (first run)
+
+### 1) Create and activate a virtual environment
+
+```bash
+python -m venv .venv
+source .venv/bin/activate
+pip install -r requirements.txt
+```
+
+### 2) Start Ollama and pull required models
+
+In a separate terminal, ensure Ollama server is running:
+
+```bash
+ollama serve
+```
+
+Then pull models:
+
+```bash
+ollama pull llama3.1
+ollama pull nomic-embed-text
+```
+
+### 3) Create a sample document
+
+```bash
+mkdir -p docs
+cat > docs/notes.txt <<'EOF'
+Paris is the capital of France.
+The Eiffel Tower is located in Paris.
+EOF
+```
+
+### 4) Ingest documents into vector DB
+
+```bash
+python app.py ingest docs/notes.txt
+```
+
+Expected output format:
+
+```text
+Ingestion complete: files=1, chunks=1
+```
+
+### 5) Start chatbot
+
+```bash
+python app.py chat
+```
+
+Example interaction:
+
+```text
+You: What is the capital of France?
+Assistant: Paris is the capital of France.
+```
+
+### 6) Run retrieval evaluation
+
+Use one or more `question::expected_substring` pairs:
+
+```bash
+python app.py eval "What is the capital of France?::Paris"
+```
+
+Expected output format:
+
+```text
+Evaluation: 1/1 hits (hit_rate=100.00%)
+```
+
+---
+
+## Helpful commands
+
+```bash
+python app.py --help
+python app.py ingest --help
+python app.py chat --help
+python app.py eval --help
+```
+
+## Troubleshooting
+
+- **Connection error to Ollama**: make sure `ollama serve` is running (default endpoint is `http://localhost:11434`).
+- **No good answers**: ingest more/better documents and increase `--top-k` in `chat` or `eval`.
+- **Import errors**: verify your virtual environment is active and dependencies are installed from `requirements.txt`.
+
+## Run tests
+
+```bash
+pytest -q
+```
