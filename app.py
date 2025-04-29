# expense_categorizer_chatbot.py
"""
Automated Expense-Categorization Chatbot  💳🤖📊
────────────────────────────────────────────────────────────────────────────
Upload your credit-card export (CSV with Date, Description, Amount, Category).
This POC:

1. Trains a simple text-classifier (Description → Category).  
2. Re-categorizes all transactions and shows a spend-by-category bar chart.  
3. Lets you download the scored CSV.  
4. Spin up a memory-enabled OpenRouter chatbot that knows your data:
   – Chat about “Why is Dining so high this month?”  
   – “Show me all Utilities Q1 vs Q2”  
   – And more, with context injected automatically.

*Demo only*—no production MLOps or compliance.  
For enterprise-grade fintech pipelines, [contact me](https://drtomharty.com/bio).
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import requests

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ───────────────────────────────────────────────────────
# OpenRouter chat helper
# ───────────────────────────────────────────────────────
API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
DEFAULT_MODEL = "mistralai/mistral-7b-instruct:free"

def send_chat(messages, model=DEFAULT_MODEL, temperature=0.7):
    if not API_KEY:
        raise RuntimeError("Set OPENROUTER_API_KEY in secrets or env")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-portfolio.com",
        "X-Title": "ExpenseCategorizerChat",
    }
    payload = {"model": model, "messages": messages, "temperature": temperature}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ───────────────────────────────────────────────────────
# Session init
# ───────────────────────────────────────────────────────
if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "system", "content":
         "You are ExpenseBot, a financial assistant. "
         "You know the user's transaction data and category-level spend."}
    ]
st.session_state.setdefault("df", None)
st.session_state.setdefault("pipeline_full", None)

# ───────────────────────────────────────────────────────
# Streamlit UI
# ───────────────────────────────────────────────────────
st.set_page_config(page_title="Expense Categorizer Chatbot", layout="wide")
st.title("💳 Expense Categorizer + Chatbot")

st.info(
    "🔔 **Demo Notice**  \n"
    "This is a proof-of-concept: simple text classifier + LLM chat.  \n"
    "For enterprise-grade solutions, [contact me](https://drtomharty.com/bio).",
    icon="💡"
)

# Upload CSV
uploaded = st.file_uploader("📂 Upload credit-card CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head())

    # Identify columns
    date_cols = df.select_dtypes(include=["object", "datetime"]).columns.tolist()
    desc_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols  = df.select_dtypes(include=["number"]).columns.tolist()

    date_col = st.selectbox("Date column", date_cols, key="date_col")
    desc_col = st.selectbox("Description column", desc_cols, key="desc_col")
    amount_col = st.selectbox("Amount column", num_cols, key="amount_col")

    if "Category" not in df.columns:
        st.error("CSV must include a 'Category' column for training.")
        st.stop()

    # Train classifier
    if st.button("🚀 Train classifier"):
        with st.spinner("Training text classifier…"):
            X = df[desc_col].astype(str)
            y = df["Category"].astype(str)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer()),
                ("clf", LogisticRegression(max_iter=1000))
            ])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            acc = accuracy_score(y_test, preds)
            st.success(f"Accuracy on hold-out: {acc:.3f}")
            st.text(classification_report(y_test, preds))
            # retrain on all data
            pipe_full = Pipeline([
                ("tfidf", TfidfVectorizer()),
                ("clf", LogisticRegression(max_iter=1000))
            ])
            pipe_full.fit(X, y)
            df["PredictedCategory"] = pipe_full.predict(X)
            st.session_state.df = df.copy()
            st.session_state.pipeline_full = pipe_full

    # If trained, show spend chart and chat
    if st.session_state.df is not None:
        df2 = st.session_state.df
        # aggregate spend
        spend = df2.groupby("PredictedCategory")[amount_col] \
                   .sum().reset_index().sort_values(amount_col, ascending=False)
        st.subheader("💰 Spend by Category")
        fig = px.bar(spend, x="PredictedCategory", y=amount_col,
                     labels={"PredictedCategory":"Category", amount_col:"Total Spend"},
                     title="Total Spend per Category")
        st.plotly_chart(fig, use_container_width=True)

        # Download scored data
        csv_bytes = df2.to_csv(index=False).encode()
        st.download_button("⬇️ Download scored CSV", csv_bytes,
                           "scored_expenses.csv", "text/csv")

        # Chat interface
        st.subheader("💬 Chat about your expenses")
        for msg in st.session_state.chat[1:]:
            st.chat_message(msg["role"]).markdown(msg["content"])

        user_input = st.chat_input("Ask me about your spend…")
        if user_input:
            st.session_state.chat.append({"role":"user","content":user_input})
            st.chat_message("user").markdown(user_input)

            # inject context: sample rows + spend summary
            sample_md = df2.head().to_markdown(index=False)
            spend_md  = spend.to_markdown(index=False)
            st.session_state.chat.append({
                "role":"system",
                "content": f"Here are your first transactions:\n\n{sample_md}"
            })
            st.session_state.chat.append({
                "role":"system",
                "content": f"Here is your spend summary by category:\n\n{spend_md}"
            })

            with st.spinner("Thinking…"):
                reply = send_chat(st.session_state.chat)
            st.session_state.chat.append({"role":"assistant","content":reply})
            st.chat_message("assistant").markdown(reply)
