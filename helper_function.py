# Importing necessary libraries and loading environment variables
import os 
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
HuggingFaceApi = os.getenv('HF_TOKEN')
groq_api_key = os.getenv("GROQ_API_KEY")

from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint

# Function to return the retrieval tool
def return_retrieval_tool():
    # Setting up the database connection and SQLDatabase
    engine = create_engine("sqlite:///AppleSales.db")
    db = SQLDatabase(engine=engine)

    # Initializing the language model
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key, temperature=1, max_tokens=1024)

    # Importing and setting up the SQLDatabaseToolkit
    from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    # Pulling the prompt template from the Langchain hub
    from langchain import hub

    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
    assert len(prompt_template.messages) == 1
    system_message = prompt_template.format(dialect="SQLite", top_k=5)

    # Importing necessary modules for agent creation
    from langchain_core.messages import HumanMessage
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver

    # Create 2nd tools to retrieve question answers from the PDF

    # Importing necessary libraries
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.tools.retriever import create_retriever_tool

    # Loading PDF data
    pdf_data = PyPDFDirectoryLoader('Pdf_data').load()

    # Splitting text from PDF
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(pdf_data)

    # Creating embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Creating vector database and retriever
    vector_database = FAISS.from_documents(text_splitter, embeddings)
    retriever = vector_database.as_retriever()

    # Creating retriever tool with prompt
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        Also, make sure to give the answer in simple English language.
        <context>
        {context}
        <context>
        Question:{input}
        """
    )

    retriever_tool = create_retriever_tool(
        retriever,
        "PDF SEARCH",
        "Search Any Kind Of Information That Is Given In The PDF Directory",
        document_prompt=prompt
    )
    combined_tools = tools + [retriever_tool] 
    
    return combined_tools
