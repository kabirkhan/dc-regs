import os
from pathlib import Path
from operator import itemgetter
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.directory import DirectoryLoader
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableSerializable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
import streamlit as st

class cited_answer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[int] = Field(
        ...,
        description="The integer IDs of the SPECIFIC sources which justify the answer.",
    )


class Citation(BaseModel):
    source_id: int = Field(
        ...,
        description="The integer ID of a SPECIFIC source which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )


class quoted_answer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )


st.set_page_config(page_title="DCRegs Housing Q&A")
st.title("Washington DC Regulations Housing Q&A")

openai_api_key = st.sidebar.text_input("OpenAI API Key")


def format_docs(docs: List[Document]) -> str:
    """Convert Documents to a single string.:"""
    formatted = [
        f"Source ID: {i}\nDocument Title: {doc.metadata['title']}\ Document Content: {doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n" + "\n\n".join(formatted)



@st.cache_resource()
def build_rag_chain(openai_api_key: str, data_dir: Path = Path("./data/housing/pdf")) -> RunnableSerializable:
    loader = DirectoryLoader(str(data_dir), glob="*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    # vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=openai_api_key))
    vectorstore = FAISS.load_local("./embeddings/title_14_housing_index/2024-02-09", OpenAIEmbeddings(api_key=openai_api_key))

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    system_msg = (
        "You're a helpful Legal AI assistant for answering user questions. "
        "Given a user question and some a set of relevant documents from the "
        "Washington DC Title 14 Housing regulations provide an answer. "
        "If none of the documents answer the question, just say I don't know.\n\n"
        "Here are the documents relevant to the user question: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human", "{question}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, api_key=openai_api_key)

    output_parser = JsonOutputKeyToolsParser(key_name="quoted_answer", return_single=True)
    llm_with_tool = llm.bind_tools(
        [quoted_answer],
        tool_choice="quoted_answer",
    )

    format = itemgetter("docs") | RunnableLambda(format_docs)
    # subchain for generating an answer once we've done retrieval
    answer = prompt | llm_with_tool | output_parser
    # complete chain that calls wiki -> formats docs to string -> runs answer subchain -> returns just the answer and retrieved docs.
    rag_chain = (
        RunnableParallel(question=RunnablePassthrough(), docs=retriever)
        .assign(context=format)
        .assign(quoted_answer=answer)
        .pick(["quoted_answer", "docs"])
    )
    return rag_chain


def generate_response(rag_chain: RunnableSerializable, prompt: str):
    res = rag_chain.invoke(prompt)
    quoted_answer = res["quoted_answer"]
    answer = quoted_answer["answer"]
    citations = quoted_answer["citations"]

    return answer, citations


prompt = st.chat_input(placeholder="Ask your question about DC Regulations related to Title 14: Housing...")
rag_chain = None
if not openai_api_key.startswith("sk-"):
    st.warning("Please enter your OpenAI API key!", icon="⚠")
else:
    rag_chain = build_rag_chain(openai_api_key=openai_api_key)
if prompt and rag_chain is not None:
    with st.chat_message("user"):
        st.write(prompt)
    answer, citations = generate_response(rag_chain, prompt)
    with st.chat_message("assistant"):
        st.write(answer)
        for c in citations:
            st.info(c["quote"])
