# bot.py
"""
Telegram Mini-RAG bot using SQLite-backed rag_engine and Hugging Face Inference API.
Commands:
/ask <question>
/summarize
/help
"""
import os
import time
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from huggingface_hub import InferenceClient

from rag_engine import MiniRAG

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Please set TELEGRAM_BOT_TOKEN in .env")

if not HF_API_KEY:
    raise ValueError("Please set HF_API_KEY in .env")

# Initialize RAG + HF
rag = MiniRAG(data_folder="data", db_path="db/embeddings.db")
hf_client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HF_API_KEY
)

# Per-user history for /summarize
user_history = {}

def add_history(uid: int, text: str):
    hist = user_history.get(uid, [])
    hist.append(text)
    if len(hist) > 3:
        hist = hist[-3:]
    user_history[uid] = hist

def call_hf_llm(prompt: str) -> str:
    try:
        res = hf_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0
        )
        return res.choices[0].message["content"].strip()
    except Exception as e:
        return f"HF API Error: {e}"

# /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Mini RAG Bot \n\n"
        "/ask <question>\n"
        "/summarize — summarize your last 3 messages\n"
        "/help"
    )

# /ask
async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /ask <question>")
        return

    query = " ".join(context.args)
    user_id = update.effective_user.id
    add_history(user_id, query)

    t0 = time.time()
    retrieved = rag.query(query, k=5)
    if not retrieved:
        await update.message.reply_text("No relevant information found.")
        return

    filtered = [r for r in retrieved if r["score"] >= 0.50]
    if not filtered:
        filtered = [retrieved[0]]

    context_txt = "\n\n".join(r["chunk"] for r in filtered)

    sources_txt = ""
    for r in filtered:
        snippet = r["chunk"][:80].replace("\n", " ")
        sources_txt += (
            f"📄 *{r['source']}* (score: {r['score']:.2f})\n"
            f"Snippet: _{snippet}..._\n\n"
        )

    prompt = f"""
Use ONLY the context below to answer.

Context:
{context_txt}

Question:
{query}

If answer isn't in context, say “I don't know”.
"""

    answer = call_hf_llm(prompt)
    total = time.time() - t0

    final_msg = (
        f"{answer}\n\n"
        f"🔍 *Sources:*\n{sources_txt}"
        f"⏱️ {total:.2f}s"
    )

    await update.message.reply_text(final_msg, parse_mode="Markdown")

# /summarize
async def summarize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    hist = user_history.get(uid, [])
    if not hist:
        await update.message.reply_text("No recent messages.")
        return

    prompt = (
        "Summarize the following in a short paragraph:\n\n"
        + "\n".join(hist)
    )
    summary = call_hf_llm(prompt)
    await update.message.reply_text(f"Summary:\n{summary}")

# run bot
if __name__ == "__main__":
    print("THE BOT IS RUNNNING..")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("summarize", summarize))

    app.run_polling()
