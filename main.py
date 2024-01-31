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

print("Processing the Files")
file_path = "/Users/I311450/Library/CloudStorage/OneDrive-SAPSE/Desktop/Tasks/api_key.txt"
file_data = open(file_path, "r")
api_key = file_data.readline()


"""# For Testing use the below two lines
print(api_key)
"""
chat = ChatOpenAI(openai_api_key=api_key)

"""
result = chat([HumanMessage(content='Most important fact about Earth')])
print(result.content)
"""

"""
Step 1: Load the Word Document 
"""
loader = Docx2txtLoader("Files/Head and Neck Cancer.docx")
doc_data = loader.load()
print("Document loaded")
print(doc_data[0].metadata)
"""
print(doc_data[0].page_content)
"""

"""
Step 2: Split the doc into chunks
"""
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
split_doc = text_splitter.split_documents(doc_data)

"""
Step 3: Convert to Vector and Persist in DB
"""
embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
db = Chroma.from_documents(split_doc, embedding_function,persist_directory="./Trial_DB")
db.persist()


print("Data Updated in DB")