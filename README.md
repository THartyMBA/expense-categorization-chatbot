# expense-categorization-chatbot

💳 Expense Categorization Chatbot
A Streamlit demo that trains a simple text classifier to auto-categorize your credit-card transactions and provides an LLM-powered chatbot to explore your spending.

Demo only—no production compliance or MLOps.
For enterprise-grade financial pipelines, contact me.

🔍 What it does
Upload a CSV of credit-card transactions with at least:

Date column

Description column

Amount column

Existing Category column for model training

Train a lightweight text classifier (TF-IDF + Logistic Regression) on your descriptions.

Re-categorize all transactions and display a spend-by-category bar chart.

Download the scored CSV (scored_expenses.csv).

Chat with a free OpenRouter LLM about your data:

“Why is Dining so high this month?”

“Show me all Utilities quarter-over-quarter”

And more, with context injected automatically.

✨ Key Features
One-click model: auto-handle text → categories

Interactive: spend charts via Plotly

Downloadable: export scored data for further analysis

Memory-enabled chat: full conversation history in st.session_state

Context injection: sample transactions + spend summary sent as hidden system messages

Zero frontend code: single Python file

🔑 Secrets
Add your OpenRouter API key so the chatbot can generate responses.

Streamlit Community Cloud
Deploy the repo → ⋯ → Edit secrets

Add:

toml
Copy
Edit
OPENROUTER_API_KEY = "sk-or-xxxxxxxxxxxxxxxx"
Local development
Create ~/.streamlit/secrets.toml:

toml
Copy
Edit
OPENROUTER_API_KEY = "sk-or-xxxxxxxxxxxxxxxx"
—or—

bash
Copy
Edit
export OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxx
🚀 Quick Start (Local)
bash
Copy
Edit
git clone https://github.com/THartyMBA/expense-categorizer-chatbot.git
cd expense-categorizer-chatbot
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run expense_categorizer_chatbot.py
Open http://localhost:8501

Upload your CSV → train classifier → view chart → chat.

☁️ Deploy on Streamlit Cloud
Push this repo (public or private) to GitHub under THartyMBA.

Go to streamlit.io/cloud → New app → select your repo/branch → Deploy.

(No additional configuration needed beyond adding your API key.)

🛠️ Requirements
shell
Copy
Edit
streamlit>=1.32
pandas
numpy
scikit-learn
plotly
requests
🗂️ Repo Structure
kotlin
Copy
Edit
expense-categorizer-chatbot/
├─ expense_categorizer_chatbot.py   ← single-file Streamlit app  
├─ requirements.txt  
└─ README.md                        ← this file  
📜 License
CC0 1.0 – public-domain dedication. Attribution appreciated but not required.

🙏 Acknowledgements
Streamlit – rapid data apps

scikit-learn – text classification

Plotly – interactive visuals

OpenRouter – free LLM gateway

Analyze your expenses, get insights, and chat away—enjoy! 🎉
