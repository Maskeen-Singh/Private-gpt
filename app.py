import streamlit as st 
import json
import os
import pinecone
import itertools
import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone

st.title("GPT")

def main():
    st.markdown("#### Input text here: ")

    # Add a text input box at the bottom of the screen
    user_input = st.text_input(" ##### **You:**", "")
    
    if user_input:
        response = f"Response: {user_input}"
        st.write(response)

if __name__ == "__main__":
    main()
    
    
st.button("**Upload**")
    
from pinecone import Pinecone
import os

os.environ["PINECONE_API_KEY"] = ["PINECONE_API_KEY"]

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

os.environ["OPENAI_API_KEY"] = ["OPENAI_API_KEY"]

from langchain_openai import OpenAIEmbeddings

#embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
embeddings = OpenAIEmbeddings()

from langchain_pinecone import PineconeVectorStore

vector_store = PineconeVectorStore(index="langchainvector", embedding=embeddings)

os.environ['OPENAI_API_KEY'] = ['OPENAI_API_KEY']
os.environ['PINECONE_API_KEY'] = ['PINECONE_API_KEY']
index_name = "langchainvector"
embeddings = OpenAIEmbeddings()
loader = TextLoader("state_of_the_union.txt", autodetect_encoding=True)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

vectorstore_from_docs = PineconeVectorStore.from_documents(
    docs,
    index_name=index_name,
    embedding=embeddings
 )
index_name = "langchainvector"
embeddings = OpenAIEmbeddings()
loader = TextLoader("India_essay.txt", autodetect_encoding=True)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

vectorstore_from_docs = PineconeVectorStore.from_documents(
    docs,
    index_name=index_name,
    embedding=embeddings
 )
def search_docs_with_history():
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) 
    docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), docsearch.as_retriever(), memory=memory)

    chat_history = []
    query = "Why NATO alliance was created?"
    result = qa.invoke({"question": query, "chat_history": chat_history})
    print("Answer 1: ", result["answer"])

    chat_history = []
    query = "The United States is a member along with how many nations?"
    result = qa.invoke({"question": query, "chat_history": chat_history})
    print("Answer 2: ", result["answer"])

    
    chat_history = []
    query = " Ford is investing $11 billion to build what?"
    result = qa.invoke({"question": query, "chat_history": chat_history})
    print("Answer 3: ", result["answer"])

    chat_history = []
    query = "The Indian constitution was adopted in which year?"
    result = qa.invoke({"question": query, "chat_history": chat_history})
    print("Answer 4: ", result["answer"])

    chat_history = []
    query = "The medieval era saw the rise of which empires?"
    result = qa.invoke({"question": query, "chat_history": chat_history})
    print("Answer 5: ", result["answer"])



search_docs_with_history()

