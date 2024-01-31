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
file_path = "/Users/I311450/Library/CloudStorage/OneDrive-SAPSE/Desktop/Tasks/api_key.txt"
file_data = open(file_path, "r")
api_key = file_data.readline()


"""# For Testing use the below two lines
print(api_key)
"""
llm = ChatOpenAI(openai_api_key=api_key,temperature=0)

"""
result = chat([HumanMessage(content='Most important fact about Earth')])
print(result.content)
"""

"""
Step 1: Load the DB Connection 
"""
embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
db = Chroma(persist_directory='./Trial_DB', embedding_function = embedding_function)


"""
Step 2: optional -> local query to return the doc only
"""
question = 'Which Trial has exclusion criteria with prosthetic'
similar_docs = db.similarity_search(question)
#print("Result of the Query")
#print(similar_docs[0].page_content)


#llm_chain = load_qa_chain(llm,chain_type='stuff')
llm_chain = load_qa_with_sources_chain(llm,chain_type='stuff', verbose=True)

result = llm_chain.run(input_documents=similar_docs,question=question)
print(result)



