# Importing necessary libraries and loading environment variables
import os 
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

load_dotenv()
HuggingFaceApi = os.getenv('HF_TOKEN')
groq_api_key = os.getenv("GROQ_API_KEY")

from helper_function import return_retrieval_tool

# Set up the Streamlit App
st.set_page_config(page_title='Chat - Bot To Interact with PDF and SQL DB')
st.title('Custom Chat-Bot')

llm = ChatGroq(model='Gemma2-9b-It', api_key=groq_api_key)

question = st.text_area("Enter Your Question", placeholder="Answer Any Question Related To The PDF Document Or A SQL Base Data?")

from langchain import hub

prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
assert len(prompt_template.messages) == 1
system_message = prompt_template.format(dialect="SQLite", top_k=5)

combined_tools = return_retrieval_tool()

from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(llm, combined_tools, messages_modifier=system_message)

def get_response(question):
    try:
        config = {"configurable": {"thread_id": "abc123"}}

        response = agent_executor.invoke(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
            config=config)
    
        return response
    except Exception as e:
        return  str(e)

question = st.text_area("Enter the question based on the PDF data and the MySQL data", "Which state has the highest number of iPhone sales?")

st.title("Question Answering App")

question = st.text_area("Enter your question:", "Which state has the highest number of iPhone sales?")

if st.button('Get Answer'):
    if question:
        with st.spinner("Generating response..."):
            # Placeholder for the actual response generation
            response = get_response(question)
            st.write("### Response:")
            st.success(response)
    else:
        st.warning("Please enter a question")
