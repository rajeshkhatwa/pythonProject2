import os
from typing import List

import chainlit.cli
import docx2txt
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
from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI

import streamlit as st

header = st.container()

with header:
    st.title('Hackathon Demo')
    docx_file = st.file_uploader("Upload Document", type=["docx"])
    if st.button("Process"):

        if docx_file is not None:
            raw_text = docx2txt.process(docx_file)
            # st.write(raw_text)
            # Schema
            schema = {
                "properties": {
                    "Title": {"type": "string"},
                    "Inclusion Criteria": {"type": "string"},
                    "Exclusion Criteria": {"type": "string"},
                    "Eligible People": {"type": "string"}
                },
                "required": ["Title"],
            }
            file_path = "/Tasks/api_key.txt"
            file_data = open(file_path, "r")
            api_key = file_data.readline()

            llm = ChatOpenAI(openai_api_key=api_key, temperature=0)
            chain = create_extraction_chain(schema, llm)
            output = chain.run(raw_text)
            st.write(output)


