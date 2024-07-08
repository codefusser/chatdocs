from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader # type: ignore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
import dotenv
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain_cohere import CohereEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from langchain.memory.summary_buffer import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_core.prompts.chat import MessagesPlaceholder

from langchain.agents import AgentExecutor, create_tool_calling_agent

import os

#from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper

dotenv.load_dotenv()
#file_path = (    "/home/hubsnippet/Downloads/papers/2205.11916v4.pdf")

#load the file to memory
#loader = PyPDFLoader(file_path)

#load the file content to data variable
#data = loader.load_and_split()

# embed the file data in a vector store

#print(data[0])

def parse_document(docs : str, question : str):
    # initialise an embedding for the vector store
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
    
    # initialise the llm
    llm = ChatCohere(model='command-r-plus')

    # split the file into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100
    )    
    docs = text_splitter.split_text(docs)

    # initialize vectorstore
    faiss_vs = FAISS.from_texts(docs, embeddings)
    # res = faiss_vs.similarity_search(input, k=2)
    llm_retriever = faiss_vs.as_retriever(llm = llm, search_kwargs={'k':1})
    res = llm_retriever.invoke(question)[0].page_content
    
    return res

#os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
#os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")
#COHERE_API_KEY = os.getenv("COHERE_API_KEY")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


# integrating an agent to perform the search with the URL
llm = ChatCohere(model="command-r-plus", cohere_api_key=COHERE_API_KEY)
    # history = MessagesPlaceholder(variable_name="history")

#question = ""
#url = ""

prompt_template = [
    ("system",  "You are a searh engine for a corpse of documentation. you will be provided with a url {url} \
    the url is your only source of information, so you should search the url pages by key words \
    you should only ground your responses with the url. \
    If {question} has no related content from the url, simple response 'no related content to your question'"),
    ("human", "{question}"),
    ("placeholder", "{agent_scratchpad}"),
    ("ai", "")]
# prompt_template = prompt_template.format(url=url, question=question)

prompt = ChatPromptTemplate.from_messages([
    #SystemMessage(content="You are a helpful assistant. You should use the google_search_name agent tool for information."),
    #HumanMessage(content="{input}"),
    #AIMessage(content="{output}"),
    ("system","You are a helpful virtual assistant." \
    "You should only use the google_search_name agent tool to search for information when necessary."),
    #MessagesPlaceholder(variable_name="history"),
    ("human","{question}"),
    ("placeholder", "{agent_scratchpad}")
])

# prompt template    
prompt_text = ChatPromptTemplate.from_messages(prompt_template)

# print(prompt_text)
# prompt template input variables
# prompt_text.input_variables = ["question", "url"], input_variables = ["question", "url"]
search = GoogleSearchAPIWrapper(google_api_key = GOOGLE_API_KEY, google_cse_id = GOOGLE_CSE_ID)

tool = Tool(
    name="google_search_name",
    description="The model should use this tool when it needs more information from the internet.",
    func=search.run,
)

agent = create_tool_calling_agent(
    tools=[tool],
    llm=llm,
    #prompt = prompt_text
    prompt = prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[tool],
    verbose=False
)

def parse_url(question : str) -> str:
    # initialise the llm 

    response = agent_executor.invoke(input = {"question": question})
    # add memmory to your conversation


    # chain your llm to prompt
    # chain = prompt_text | llm | StrOutputParser()
    # chain = conversation_llm | StrOutputParser()

    #response = chain.invoke(input = {"question" : question, "url":url})
    return response

# message = HumanMessage(content="inurl: https://learn.microsoft.com 'what are cloud security best practices'")

# print(parse_url(message))
