import os
import streamlit as st
from dotenv import load_dotenv
from triage import run_triage

load_dotenv()
st.set_page_config(page_title="Fintech Query Triage", page_icon="💳")
st.title("💳 Fintech Customer Query Triage Assistant")
st.caption("LangChain prompts • Pydantic validation • parsers • chains • runnables")
st.warning("Educational demo only. Do not enter real customer data, account details, credentials, OTPs, PINs, or CVVs.")

with st.sidebar:
    mode = st.radio("Mode", ["Demo mode (no API key)", "OpenAI mode"])
    st.write("OpenAI mode requires OPENAI_API_KEY in your private .env file.")

examples = {
    "Loan repayment": "I paid my loan installment yesterday, but the app still shows it as unpaid.",
    "Payment failure": "My payment failed but money was deducted from my bank account.",
    "Loan statement": "I need a copy of my loan statement for the previous financial year.",
    "Account access": "I cannot log in after resetting my password.",
    "Possible fraud": "I noticed an unauthorized transaction on my account."
}
choice = st.selectbox("Example", ["Write my own"] + list(examples))
message = st.text_area("Customer message", examples.get(choice, ""), height=130)
context = st.text_input("Optional non-sensitive context", placeholder="e.g. product type or support channel")

if st.button("Triage query", type="primary", use_container_width=True):
    if not message.strip():
        st.error("Enter a customer message.")
    elif mode == "OpenAI mode" and not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is missing. Configure .env or select Demo mode.")
    else:
        try:
            result = run_triage(message, context, mode == "OpenAI mode")
            c = result["classification"]
            a, b, d = st.columns(3)
            a.metric("Category", c["category"].replace("_", " ").title())
            b.metric("Urgency", c["urgency"].title())
            d.metric("Confidence", f'{c["confidence"]:.0%}')
            st.subheader("Summary")
            st.write(c["summary"])
            st.subheader("Suggested queue")
            st.info(result["route"]["queue"])
            st.write(result["route"]["reason"])
            st.subheader("Draft reply — human review required")
            st.text_area("Draft", result["draft_reply"], height=120)
            with st.expander("Validated structured output (JSON)"):
                st.json(c)
            with st.expander("Parallel analysis"):
                st.json(result["parallel_analysis"])
            st.caption("No account was accessed and no action or message was sent.")
        except Exception as exc:
            st.error(f"Could not process query: {exc}")
