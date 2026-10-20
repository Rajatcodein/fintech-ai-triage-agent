# Fintech Customer Query Triage Assistant

An educational portfolio project demonstrating LangChain prompt templates, Pydantic structured output validation, output parsers, chains, `RunnableParallel`, `RunnableBranch`, and `RunnableLambda`.

**Demo only:** it does not access bank systems, perform transactions, make financial decisions, or send replies. Never enter real customer data, account numbers, credentials, OTPs, PINs, or CVVs.

## Features
- Dynamic prompts using customer message and optional context
- Pydantic model with `Literal` types and field constraints
- `PydanticOutputParser` to parse and validate classification
- `StrOutputParser` to extract reply text
- Pipe-composed LangChain chains
- Parallel metadata checks and conditional routing
- Human-reviewed draft reply
- Streamlit interface and no-key demo mode

## Architecture
```text
Customer message -> ChatPromptTemplate -> LLM -> PydanticOutputParser
                                               |
                          RunnableParallel -----+
                                               |
                          RunnableBranch -> Suggested support queue
                                               |
                          Reply prompt -> LLM -> StrOutputParser
                                               |
                                   Draft for human review
```

## Run locally (Windows / VS Code)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
If PowerShell activation is blocked, use Command Prompt: `.venv\Scripts\activate.bat`.

## Optional OpenAI mode
1. Copy `.env.example` to `.env`.
2. Add your API key to `.env` (never commit or share it).
3. Run `streamlit run app.py` and choose OpenAI mode. This may incur provider API costs.
4. Demo mode works without an API key.

## Concepts mapped to code
- Prompt templates: `classification` and `reply` prompts in `triage.py`
- Structured output: `QueryClassification` Pydantic model
- Output parsers: `PydanticOutputParser`, `StrOutputParser`
- Chains: `prompt | llm | parser`
- Parallel: `RunnableParallel`
- Conditions: `RunnableBranch`
- Reusable functions: `RunnableLambda`

## Limitations and safety
Pydantic validates shape and types, not truth. The demo classifier is rule-based and illustrative; its confidence values are not calibrated. Real deployment requires approved policies, authorization, audit controls, privacy safeguards, testing, monitoring, and human escalation. Do not automatically send drafts or take account actions.

## Publish to GitHub
Create a new repository, e.g. `fintech-customer-query-triage`, then run these commands from this folder:
```bash
git init
git branch -M main
git add .
git status
git commit -m "Build LangChain fintech query triage assistant"
git remote add origin https://github.com/YOUR_USERNAME/fintech-customer-query-triage.git
git push -u origin main
```
Replace `YOUR_USERNAME` with your GitHub username. Confirm `.env` is not staged before pushing.
