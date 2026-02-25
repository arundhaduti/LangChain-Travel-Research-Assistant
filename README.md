# Travel Research Assistant – RAG with LangChain and Pydantic

This repository demonstrates how to build a simple Retrieval Augmented Generation (RAG) application using:

- LangChain (LCEL style)
- Local HuggingFace embeddings
- OpenRouter for LLM inference
- Pydantic for structured validation
- FAISS vector database

## Use Case

A **Travel Research Assistant** that can generate structured travel itineraries based on a small internal knowledge base.

Example query:

> "Create a 4 day Bengaluru itinerary under 12000 rupees"

The system retrieves relevant information from local documents and returns a validated JSON travel plan.

---

## Features

- Fully local embeddings using `sentence-transformers`
- No dependency on OpenAI
- Uses OpenRouter models for flexible LLM usage
- Strong output validation with Pydantic
- Clean and modular LangChain implementation

---

## Project Structure

```
travel-rag/
│
├── app.py
├── models.py
├── data/
│   ├── mumbai.txt
│   └── bengaluru.txt
├── .env
└── README.md
```

---

## Installation

### 1. Clone the repository

```
git clone <your-repo-url>
cd travel-rag
```

### 2. Install dependencies

```
uv pip install langchain-core langchain-community langchain-openai langchain-text-splitters langchain-huggingface sentence-transformers faiss-cpu python-dotenv
```

### 3. Configure Environment Variables

Create a `.env` file inside the `travel-rag` folder:

```
OPENROUTER_API_KEY=your_openrouter_api_key
```

---

## Running the Application

```
python app.py
```

Example Output:

```
destination='Bengaluru'
total_days=4
total_budget=12000
plans=[...]
```

---

## How It Works

1. Text documents are loaded from the `data/` directory.
2. They are split into chunks.
3. Local HuggingFace embeddings convert them into vectors.
4. FAISS stores and retrieves relevant chunks.
5. OpenRouter LLM generates an answer.
6. Pydantic validates the response structure.

---

## Customization

- Add more `.txt` files in the `data/` folder to expand knowledge.
- Change the OpenRouter model in `app.py`.
- Modify `models.py` to enforce stricter validation.

---

## Technologies Used

- Python
- LangChain
- FAISS
- HuggingFace Sentence Transformers
- OpenRouter
- Pydantic

---

## License

MIT
