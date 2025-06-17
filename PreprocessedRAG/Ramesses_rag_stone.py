from langchain_community.document_loaders import PyPDFLoader

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

from langchain.retrievers import MultiQueryRetriever

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables import RunnablePassthrough


from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import streamlit as st

import os
import re

from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def preprocess_text(text):
    #hyphenated words broken by newlines
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    #specific artifacts we know are problematic
    text = re.sub(r'\ba\)\s*[-.]*\s*', '', text)  # Remove 'a) - .-' patterns
    text = re.sub(r'\btel\s*\|\s*\d+\b', '', text)  # Remove 'tel | 7' patterns
    text = re.sub(r'\{[^}]*\}', '', text)  # Remove anything in curly braces
    text = re.sub(r'\(\s*(see|fig|figure)[^)]*\)', '', text, flags=re.I)  # Remove figure references
    #noticed problematic characters with space
    text = re.sub(r'[={}<>\|~©•†‡_]', ' ', text)
    #isolated garbage characters (not between words)
    text = re.sub(r'(?<!\w)[-.]+\s*', '', text)  # Remove leading hyphens/dots
    text = re.sub(r'\s+[-.]+(?!\w)', '', text)  # Remove trailing hyphens/dots
    #whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_clean_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    cleaned_texts = []
    for doc in docs:
        text = doc.page_content
        text = preprocess_text(text)
        #fixes for common PDF artifacts
        text = re.sub(r'^\W+', '', text)
        text = re.sub(r'\s+-\s+', '-', text)  
        cleaned_texts.append(text)
    return "\n\n".join(cleaned_texts)

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
    valid_documents = []
    for doc in documents:
        # Ensure page_content is a non-empty string
        if not hasattr(doc, 'page_content') or not isinstance(doc.page_content, str) or not doc.page_content.strip():
            print(f"Skipping invalid document: {doc.metadata}")
            continue
        doc.metadata = {k: v for k, v in doc.metadata.items() if v}
        if "page" in doc.metadata:
            doc.metadata["source"] = f"Page {doc.metadata['page']}"
        else:
            doc.metadata["source"] = "Unknown"
        valid_documents.append(doc)
    return valid_documents

def upload_data(pdf_path="the_life_and_times_of_Ramesses_II.pdf"):
    """One-time function to process PDF and upload to Qdrant."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    cleaned_text = load_and_clean_pdf(pdf_path)
    chunked_text = split_text(cleaned_text)
    documents = preprocess_documents(chunked_text)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Load credentials from environment
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = "new_ramesses_ii_docs"

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=True
    )

    # Delete existing collection to ensure a clean upload
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception as e:
        print(f"No existing collection to delete: {e}")

    print(f"Uploading {len(documents)} chunks to Qdrant...")

    vectorstore = Qdrant.from_documents(
        documents=documents,
        embedding=embedding_model,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        prefer_grpc=True
    )
    print("✅ Upload successful.")
    
    

def get_retriever():
    """Connect to existing Qdrant collection and return retriever."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Load from .env
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = "new_ramesses_ii_docs"

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=True
    )

    try:
        collection_info = client.get_collection(collection_name)
        if collection_info.vectors_count == 0:
            raise Exception("Qdrant collection is empty. Run upload_data() first.")
        print("✅ Using existing Qdrant collection.")
        vectorstore = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embedding_model
        )
        return vectorstore
    except Exception as e:
        raise Exception(f"Qdrant collection not found or empty: {e}. Run upload_data() first.")
    

    
def onChain():
    
    # Add this at the beginning of your onChain() function
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=5, 
            return_messages=True,
            memory_key="chat_history"
        )
    # Load Google API key from environment variable
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

    # Create or retrieve memory from Streamlit's session state
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=5, 
            return_messages=True, 
            memory_key="chat_history"
        )
    
    # Get Qdrant retriever
    vector_retriever = get_retriever()

    multi_query_prompt = PromptTemplate.from_template("Generate multiple versions of this question: {question}")

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=vector_retriever.as_retriever(search_kwargs={"k": 5}),
        llm=llm,
        prompt=multi_query_prompt
    )

    # Same prompt template as before
    prompt_template = """
    You are Ramesses II, Pharaoh of Egypt, a divine ruler known for your wisdom, strength, and connection to the gods.
    You must respond in **first-person** as Ramesses II, addressing the user as a visitor from distant lands.

    ### **Response Guidelines:**
    - **Be immersive**: Speak as a Pharaoh would-use grand, poetic language.
    - **Use storytelling**: Explain events from your own experiences.
    - **Be interactive**: Ask the user follow-up questions to keep the conversation engaging.

    ### **Tone & Style:**
    - Your words should feel **royal yet welcoming**.
    - Use **historical details** and **authentic phrases**.

    ### **Rules:**
    1. **Never break character**. You ARE Ramesses II.
    2. **Base answers ONLY on context**. If unsure, admit it in a regal, humorous way.
    3. **Keep responses under 200 words** to maintain engagement.

    ---
    
    ### **Chat History:**
    {chat_history}
    
    ### **Context:**
    {context}

    ### **User's Question:**
    {input}

    ### **Your Response (as Ramesses II)**:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Create the chain
    chain = (
        {"context": multi_query_retriever | format_docs, 
         "input": RunnablePassthrough(),
         "chat_history": lambda x: st.session_state.memory.load_memory_variables({})["chat_history"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Create a function that handles memory updates
    def chain_with_memory(user_input):
        result = chain.invoke(user_input)
        st.session_state.memory.save_context({"input": user_input}, {"output": result})
        return result
    
    return chain_with_memory
