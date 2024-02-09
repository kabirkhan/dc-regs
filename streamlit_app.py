import streamlit as st
from langchain.llms import OpenAI


st.set_page_config(page_title="DCRegs Housing Q&A")
st.title("Washington DC Regulations Housing Q&A")

openai_api_key = st.sidebar.text_input("OpenAI API Key")

def generate_response(input_text: str):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    st.info(llm(input_text))

with st.form("qa"):
    text = st.text_area("Question:", placeholder="Ask your question about DC Regulations related to Title 14: Housing...")
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-"):
        generate_response(text)
