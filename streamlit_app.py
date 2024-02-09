import os
from pathlib import Path
from operator import itemgetter

from langchain.llms import OpenAI
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.directory import DirectoryLoader
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableSerializable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st


st.set_page_config(page_title="DCRegs Housing Q&A")
st.title("Washington DC Regulations Housing Q&A")

openai_api_key = st.sidebar.text_input("OpenAI API Key")


def build_rag_chain(openai_api_key: str, data_dir: Path = Path("./data/housing/pdf")) -> RunnableSerializable:
    loader = DirectoryLoader(str(data_dir), glob="*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    return rag_chain


def generate_response(rag_chain: RunnableSerializable, input_text: str):
    res = rag_chain.invoke(input_text)

    st.info(res["question"])
    st.info(res["answer"])
    st.info("Sources:")
    st.info("\n".join(set([doc.metadata["file_path"].strip() for doc in res["context"]])))

with st.form("qa"):
    text = st.text_area("Question:", placeholder="Ask your question about DC Regulations related to Title 14: Housing...")
    submitted = st.form_submit_button("Submit")
    rag_chain = None
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    else:
        rag_chain = build_rag_chain(openai_api_key=openai_api_key)
    if submitted and rag_chain is not None:
        generate_response(rag_chain, text)
