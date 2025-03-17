from jinja2 import Environment
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter  # ✅ Fixed Import

#from langchain_text_splitters import SpacyTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import MultiQueryRetriever

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.prompts import ChatPromptTemplate
#from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain.agents import create_openai_functions_agent, AgentExecutor
import os
import glob
from langchain.llms import BaseLLM
import requests
import re

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

#from langchain.embeddings import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer

#from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM  
#from langchain_openai import ChatOpenAI

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever  

# -----| PREPROCESSING |----------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def preprocess_text(text):
    text = re.sub(r"\n+", "\n", text)  
    text = re.sub(r"-\n", "", text)  
    text = re.sub(r"\(see.*?figure.*?\)", "", text, flags=re.I)  
    return text.strip()

def load_and_clean_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    cleaned_text = "\n\n".join([preprocess_text(doc.page_content) for doc in docs])
    return cleaned_text

def split_text(cleaned_text):    
    headers_to_split_on = [
        ("CHAPTER", "chapter"),
        ("PART", "part"),
        ("Section", "section")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    section_chunks = markdown_splitter.split_text(cleaned_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=100  
    )
    final_chunks = text_splitter.split_documents(section_chunks)
    
    return final_chunks

def preprocess_documents(documents):
    for doc in documents:
        doc.metadata = {k: v for k, v in doc.metadata.items() if v}
        
        if "page" in doc.metadata:
            doc.metadata["source"] = f"Page {doc.metadata['page']}"
        else:
            doc.metadata["source"] = "Unknown"

    return documents

def store_in_chroma(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"  
    )
    
    return vector_store.as_retriever()
def onChain():
    #llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="AIzaSyDH6MKdZLY5DFwPGu2YbzRsah2mrC9LLq4")
    #llm = Ollama(model="llama2")
    llm = OllamaLLM(model="llama2")

    pdf_path = "the_life_and_times_of_Ramesses_II.pdf"
    
    cleaned_text = load_and_clean_pdf(pdf_path)
    
    documents = split_text(cleaned_text)

    documents = preprocess_documents(documents)

    vector_retriever = store_in_chroma(documents)

    bm25_retriever = BM25Retriever.from_documents(documents)
    
    multi_query_prompt = PromptTemplate.from_template("Generate multiple versions of this query: {query}")
    multi_query_chain = multi_query_prompt | llm
    multi_query_retriever = vector_retriever | RunnablePassthrough() | multi_query_chain
    
    prompt_template = """
    You are Ramesses II, Pharaoh of Egypt, a divine ruler known for your wisdom, strength, and connection to the gods. 
    You must respond in **first-person** as Ramesses II, addressing the user as a visitor from distant lands.

    ### **Response Guidelines:**
    - **Be immersive**: Speak as a Pharaoh would—use grand, poetic language.
    - **Use storytelling**: Explain events from your own experiences.
    - **Be interactive**: Ask the user follow-up questions to keep the conversation engaging.

    ### **Tone & Style:**
    - Your words should feel **royal yet welcoming**.
    - Use **historical details** and **authentic phrases**.
    - Describe **body language and gestures** inside double asterisks (**like this**).

    ### **Rules:**
    1. **Never break character**. You ARE Ramesses II.
    2. **Base answers ONLY on context**. If unsure, admit it in a regal, humorous way.
    3. **Keep responses under 200 words** to maintain engagement.

    ---

    ### **Context:**
    {context}

    ### **User's Question:**
    {input}

    ### **Your Response (as Ramesses II)**:
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    chain = (
        {"context": multi_query_retriever | bm25_retriever | format_docs, "input": RunnablePassthrough()}  # ✅ Fixed `retriever`
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

