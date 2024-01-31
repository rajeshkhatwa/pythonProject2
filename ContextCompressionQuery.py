import os
import openai
import chromadb
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import Docx2txtLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import logging

logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

"""
pip install langchain
openAI
doc2txt
chromadb
tiktoken
"""
from langchain.output_parsers import DatetimeOutputParser
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

print("Hi dff")
file_path = "Tasks/api_key.txt"
file_data = open(file_path, "r")
api_key = file_data.readline()

llm = ChatOpenAI(openai_api_key=api_key, temperature=0)

"""
Step 1: Load the DB Connection 
"""
embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
db = Chroma(persist_directory='./Trial_DB', embedding_function=embedding_function)

"""
Step 2: Use Compression
"""
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                       base_retriever=db.as_retriever())

question = 'Which are the trials done in Egypt. Provide  Government identifier as well'

compressed_doc = compression_retriever.get_relevant_documents(question)

print("query Output")

print(compressed_doc[0].page_content)
