import os
from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    file_path = "/Users/I311450/Library/CloudStorage/OneDrive-SAPSE/Desktop/Tasks/api_key.txt"
    file_data = open(file_path, "r")
    api_key = file_data.readline()
    print("Inside On Chat Start")

    llm = ChatOpenAI(openai_api_key=api_key, temperature=0,model_name='gpt-3.5-turbo', streaming=True)
    """
    Step 1: Load the DB Connection 
    """
    embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
    db = Chroma(persist_directory='./Trial_DB', embedding_function=embedding_function)

    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,

    )
    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        memory=memory,
        verbose=True,
        return_source_documents=True,
    )
    msg = cl.Message(content="Hello Good Morning")
    await msg.send()
    msg.content = "You are giving Demo for the AI Life Sciences Hackathon. Ask your Question. I will make sure your demo is successful"
    await msg.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    res = await cl.make_async(chain)(message.content, callbacks=[cb])
    answer = res["answer"]
    print(answer)

    await cl.Message(content=answer).send()
