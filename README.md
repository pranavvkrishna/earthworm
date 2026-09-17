# Earthworm AI

An agentic AI assistant for farmers that answers questions about crop diseases, USDA programs, and weather — through natural conversation, powered by an LLM agent that decides which tool to use.

Instead of separate apps for each need, Earthworm routes a single question to the right specialist: a computer vision model for disease diagnosis, a RAG pipeline grounded in real USDA documents, or a live weather API — and returns one grounded answer.

## Features

- **Crop Disease Diagnosis** — Fine-tuned ResNet18 (transfer learning) classifies 15 disease categories across tomato, potato, and pepper plants. Trained on the PlantVillage dataset, ~98% validation accuracy.
- **USDA Program Navigator** — Retrieval-augmented generation over real USDA fact sheets (crop acreage reporting, disaster assistance, payment eligibility, beginning farmer loans). Answers are grounded in retrieved source documents and cite where the information came from — the system explicitly says when it doesn't know, rather than guessing.
- **Weather Lookup** — Live forecast data via geocoding + the National Weather Service API.
- **Agentic Routing** — A LangGraph-based agent reads each question and decides which tool (if any) to call, so the user never has to specify which "mode" they want.

## How It Works

```
User question
      |
   LLM Agent (planner)
      |
  +---+----------------+--------------+
  |                     |              |
Vision Tool         RAG Tool      Weather Tool
(ResNet18)      (ChromaDB + LLM)  (NWS API)
  |                     |              |
  +---------+-----------+--------------+
            |
    Grounded answer back to user
```

## Tech Stack

| Layer | Tools |
|---|---|
| Agent orchestration | LangGraph, LangChain |
| Computer vision | PyTorch, torchvision (ResNet18, transfer learning) |
| RAG | ChromaDB, Sentence-Transformers (local embeddings), PyPDF |
| LLM | Ollama (local) |
| Weather | Open-Meteo (geocoding), National Weather Service API |
| Backend | Python, FastAPI |

## Project Structure

```
earthworm/
├── backend/
│   └── app/
│       ├── agent/       # LangGraph agent + tool routing
│       ├── rag/          # Ingest, retrieval, and generation pipeline
│       ├── vision/       # CV model inference
│       └── weather/      # Weather lookup tool
├── data/
│   └── usda_docs/         # Source PDFs for the RAG knowledge base
└── notebooks/              # Data exploration
```

## Setup

```bash
# Clone and set up environment
git clone https://github.com/pranavvkrishna/earthworm.git
cd earthworm/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Pull the local LLM
ollama pull llama3.2

# Build the RAG knowledge base
cd app/rag
python ingest.py

# Run the agent
cd ../agent
python agent.py
```

## Known Limitations

- **Domain mismatch in vision model**: The CV model was trained on the PlantVillage dataset, which consists of single, isolated leaf photos on clean backgrounds. It performs strongly on similarly-framed images but is less reliable on full-plant, cluttered field photos — a common and well-documented challenge when deploying benchmark-trained CV models in real-world conditions. A production version would add a detection/cropping step to isolate individual leaves before classification.
- **Small RAG knowledge base**: Currently indexed on a handful of USDA fact sheets. Retrieval quality and coverage would improve with a larger, more diverse document set.
- **Local LLM reasoning**: The agent's planning/synthesis LLM runs locally via Ollama for zero-cost iteration. To avoid hallucination during multi-step synthesis, single-tool responses are returned directly rather than re-summarized by the LLM.

## Roadmap

- [ ] FastAPI endpoints + frontend chat UI
- [ ] Conversation memory across turns
- [ ] Web search tool for questions outside the RAG knowledge base
- [ ] Leaf-detection/cropping preprocessing step for real-world photos
