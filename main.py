import streamlit as st
import os
from langchain.embeddings import OpenAIEmbeddings
import qdrant_client
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain , RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from getpass import getpass
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank 
import cohere
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import time 
import asyncio
import regex as re


# Initialize Qdrant client and create collections
qdrant_client = QdrantClient(
    url=os.environ['QDRANT_URL']#"
    api_key=os.environ['QDRANT_API_KEY']#
os.environ["QDRANT_COLLECTION_NAME"] = "website_conversations_v1"  # unique
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
vector_store = Qdrant(
    client=qdrant_client,
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    embeddings=embeddings
)

# Initialize language model and conversational retrieval chain
llm = ChatOpenAI(
    temperature=0.2,
    model="gpt-4-1106-preview",
    openai_api_key=os.environ['OPENAI_API_KEY']
)

compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 5})
)

async def get_relevent_docs(query):
    res =  await compression_retriever.aget_relevant_documents(query=user_input)
    _ = await asyncio.sleep(1)
    return res

def get_response(context , query:str):
    res=llm.predict(f"""
You are a helpful assistant. Your Job is to read context and answer to user. While writing answer make sure which page content you got a answer you have to write that Source link at the end of that line. 
For Example:
This answer get from first page content. Source link: https://****.
This answer comes from 3rd page content. Source link: https://****.

### Context:
{context}

### User Message:
{query}

### Answer:
""")
    
    return res


# Streamlit interface
st.title("Medical Bot 🤖")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

context=""
# Accept user input
if user_input := st.text_input("Ask me a question:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner(text="processing..."):

        result = asyncio.run(get_relevent_docs(user_input))
        
        for data in result:
            meta_data = data.metadata['source']
            chunks  = meta_data.split('\\')
            pdf_name = chunks[5][:-4]
            base_url= f'https://www.ncbi.nlm.nih.gov/books/{pdf_name}/'
            page_content = data.page_content
            context+=f"Source link: {base_url} \n page_content: {page_content}"

        answer = get_response(context , query=user_input)
        pattern = r'https?:\S+'
        urls = re.findall(pattern=pattern , string=answer)#search(pattern, answer)
        
        
        with st.chat_message("assistant"):
            assistant_response = answer
            for idx, url in enumerate(urls, start=1):
                clickable_url = f"[[{idx}]]({url})"
                assistant_response = assistant_response.replace(url, clickable_url)
            st.markdown(assistant_response)
            
