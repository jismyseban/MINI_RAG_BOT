# Mini-RAG Telegram Bot

A lightweight Retrieval-Augmented Generation (RAG) bot built for Telegram.  
It retrieves answers from a small domain-specific knowledge base using local embeddings  
and generates responses using Hugging Face Inference API.

---


### ✔ Mini-RAG
- Uses 3–5 text/markdown documents from `data/`
- Splits documents into ~150-word chunks
- Embeds using **all-MiniLM-L6-v2**
- Stores embeddings in **SQLite (`db/embeddings.db`)**
- Retrieves top-k similar chunks (cosine similarity)
- Constructs RAG prompt and sends to HuggingFace Inference API
- Responds via Telegram `/ask` command
- Message history awareness (last 3 messages)
- Query-level caching
- Source snippets + similarity scores
- `/summarize` command for summarizing recent user messages
- Filters sources by score ≥ **0.50** (fallback to top-1)
- Incremental indexing using file SHA1

### ✔ Telegram Commands
```
/ask <question>
/summarize
/help
```
---

# 🧠 Tech Stack
```
| Component | Tech |
|----------|------|
| Bot Framework | python-telegram-bot |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| LLM Backend | Hugging Face Inference API (Mistral-7B-Instruct) |
| Language | Python 3.10+ |
| Storage | SQLite + query caching |
```

---

# 🗂 Folder Structure
```
RAG_BOT/
├── bot.py
├── rag_engine.py
├── requirements.txt
├── README.md
│
├── data/
│ ├── company_policy.txt
│ ├── reimbursement_policy.txt
│ ├── work_hours.txt
│ ├── leave_policy.txt
│ └── onboarding_guide.txt
│
├── .env
├── db/
│ ├── embeddings.db
```

---

# ⚙️ Setup Instructions

### 1. Create virtual environment (Windows)
```
python -m venv rag_env
rag_env\Scripts\activate # Windows
```

### 2. Install dependencies
```
pip install -r requirements.txt
```


### 3. Add environment variables  
```
Create a file named `.env`:

TELEGRAM_BOT_TOKEN=your-telegram-token
HF_API_KEY=hf_xxxxxxxxxxxxx
```
---

### 4. Running the bot

```
python bot.py
```

You should see:
` THE BOT IS RUNNNING.. `


Now open Telegram → search for **@Grawp_Bot** → and test:




## Example prompts
```
/ask What is the reimbursement procedure?
/ask What are the working hours?
/ask How do I reset my laptop password?
/ask What do I need on the first day?
/ask What is quantum entanglement?
/summarize
/help
```





