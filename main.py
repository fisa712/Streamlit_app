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

os.environ["COHERE_API_KEY"] = "zbIa9oYfhPZb1lWnvwxNpk9bN1cP1GLhdughs1zd"

# Initialize Qdrant client and create collections
qdrant_client = QdrantClient(
    url="https://02748712-9e97-4644-86bc-47d79248dffb.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="EalPrnSaJH-LSTNfAPTMTwXdPuLahsu_JlqKfKUM_-aVrDk8Ao8JEA",
)
os.environ["QDRANT_COLLECTION_NAME"] = "website_conversations_v1"  # unique
embeddings = OpenAIEmbeddings(openai_api_key="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")
vector_store = Qdrant(
    client=qdrant_client,
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    embeddings=embeddings
)

# Initialize language model and conversational retrieval chain
llm = ChatOpenAI(
    temperature=0.2,
    model="gpt-4",
    openai_api_key="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"
)

compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 3})
)

qa_chain = RetrievalQA.from_chain_type(
    chain_type='stuff' ,
    llm=llm,
    retriever=compression_retriever,
    return_source_documents=True
)

# Streamlit interface
st.title("Medical Bot 🤖")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Accept user input
if user_input := st.text_input("Ask me a question:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner(text="processing..."):
        # Query the conversational retrieval chain
        result = qa_chain({'query': user_input})

        # Extract metadata and generate links
        metadata = [doc.metadata['source'] for doc in result['source_documents']]
        paths = list(set(metadata))
        sources = paths[0].split('\\')[5]
       
        links = f'https://www.ncbi.nlm.nih.gov/books/{sources[:-4]}/'
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            assistant_response = f"\n{result['result']}\n\nSource: {links}"
            st.markdown(assistant_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
