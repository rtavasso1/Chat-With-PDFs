import streamlit as st
import pandas as pd
import os
import base64
from langchain.document_loaders import PyPDFLoader
import os


st.title("ChatGPT with Document Query")

# Read the files from the directory
directory = r"./example-docs"
files = []
if directory:
    for file in os.listdir(directory):
        if file.endswith(".txt") or file.endswith(".pdf") or file.endswith(".docx") or file.endswith(".xlsx"):
            files.append(file)
st.write("Uploads are unavailable. The available files are listed below.")

# Add a way to select which files to use for the model query
selected_files = st.multiselect("Please select the files to query:", options=files)

def process_files(selected_files):
    for file in selected_files:
        # add directory to file
        file = os.path.join(directory, file)
        loader = PyPDFLoader(file)
        document = loader.load()
        st.write(document)
    return

if st.button("Process Files"):
    process_files(selected_files)
def model_query(query, document_names):
    response = f"Sample response for the query '{query}' over the documents: {', '.join(document_names)}"
    return response

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question over the selected documents:")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    model_response = model_query(user_input, selected_files)
    st.session_state.chat_history.append({"role": "model", "content": model_response})
    user_input = ""

for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"> **User**: {message['content']}")
    else:
        st.markdown(f"> **Model**: {message['content']}")
