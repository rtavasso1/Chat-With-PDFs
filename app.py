import streamlit as st
import os
from langchain.vectorstores import FAISS
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import glob
import pickle
from pathlib import Path
import hashlib
import tempfile
import shutil
import logging
logging.basicConfig(level=logging.INFO)

st.title("ChatGPT with Document Query")


# Define necessary embedding model, LLM, and vectorstore
openai_api_key = st.secrets["OPENAI_API_KEY"]
text_key = "text"

def hash_content(content):
    return hashlib.md5(content.encode('utf-8')).hexdigest()

@st.cache_resource()
def load_vectorstore(buffers=None):
    example_docs_path = Path("example-docs")
    filenames = list(example_docs_path.glob("*.pdf"))

    loaders = [PyPDFLoader(str(filename)) for filename in filenames]
    listOfPages = [loader.load_and_split() for loader in loaders]

    listOfBufferPages = []
    buffer_names = []

    if buffers:
        # Create temporary files for buffers and load them
        for buffer in buffers:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            buffer_names.append(buffer.name)
            with open(temp_file.name, 'wb') as f:
                f.write(buffer.getbuffer())  # Write buffer contents to a temporary file
            buffer_loader = PyPDFLoader(temp_file.name)
            bufferPages = buffer_loader.load_and_split()
            for page in bufferPages:
                page.metadata['source'] = buffer.name
            listOfBufferPages.append(bufferPages)
            temp_file.close()  # Close the file so PyPDFLoader can access it

    all_pages = listOfPages + listOfBufferPages if buffers else listOfPages
    all_names = filenames + buffer_names if buffers else filenames

    # Create FAISS indices for all files and buffers
    faiss_indices = {
        str(name): FAISS.from_documents(pages, OpenAIEmbeddings(disallowed_special=()))
        for name, pages in zip(all_names, all_pages)
    }

    # Clean up temporary files
    for name in buffer_names:
        Path(name).unlink(missing_ok=True)

    return faiss_indices

faiss_indices = load_vectorstore()

def initialize_conversation():
    chat = ChatOpenAI(model_name=model_version, temperature=0)
    template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. Excerpts from relevant documents the AI has read are included in the conversation and are used to answer questions more accurately. The AI is not perfect, and sometimes it says things that are inconsistent with what it has said before. The AI always replies succinctly with the answer to the question, and provides more information when asked. The AI recognizes questions asked to it are usually in reference to the provided context, even if the context is sometimes hard to understand, and answers with information relevant from the context.

    Current conversation:
    {history}
    Friend: {input}
    AI:"""
    PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=template
    )
    st.session_state.conversation = ConversationChain(
        prompt=PROMPT,
        llm=chat, 
        verbose=False, 
        memory=ConversationBufferMemory(human_prefix="Friend", )
    )


# Read the files from the directory using pathlib
directory = Path("./example-docs")
files = []
if directory.exists():
    for file in directory.iterdir():
        if file.suffix in [".txt", ".pdf", ".docx", ".xlsx"]:
            files.append(str(file))

def getTopK(query, doc_path):
    doc_name = str(Path(doc_path))  # Normalize and convert to string
    related = faiss_indices[doc_name].similarity_search_with_relevance_scores(query, k=num_sources)
    return related

def model_query(query, document_names):
    # Gather related context from documents for each query
    all_related = []
    for document_name in document_names:
        related = getTopK(query, document_name)
        all_related.extend(related)
    all_related = sorted(all_related, key=lambda x: x[1], reverse=True)

    # Filter out the context excerpts already present in the conversation and check relevancy score
    unique_related = []
    context_count = 0
    MIN_RELEVANCY_SCORE = 0.0

    for r in all_related:
        if context_count >= num_sources:
            break
        content_hash = hash_content(str(r[0].page_content))
        logging.info("Score: %s", r[1])
        if content_hash not in st.session_state.context_hashes and r[1] >= MIN_RELEVANCY_SCORE:
            unique_related.append(r)
            st.session_state.context_hashes.add(content_hash)
            context_count += 1

    related = [r[0] for r in unique_related]

    if not related:
        ai_message = st.session_state.conversation.predict(input=query)
        return ai_message, None
    else:
        context = " ".join([r.page_content for r in related])
        ai_message = st.session_state.conversation.predict(input=context + " " + query)
        return ai_message, related
    
# Sidebar elements for file uploading and selecting options
with st.sidebar:
    st.title("Document Query Settings")

    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
    if uploaded_files:
        # Reload the vectorstore with new files including uploaded ones
        faiss_indices = load_vectorstore(buffers=uploaded_files)  

        uploaded_file_names = [uploaded_file.name for uploaded_file in uploaded_files]
        files.extend(uploaded_file_names)

    # Add a way to select which files to use for the model query
    selected_files = st.multiselect("Please select the files to query:", options=files)

    # Add a slider for number of sources to return 1-5
    num_sources = st.slider("Number of sources per document:", min_value=1, max_value=5, value=1)

    model_version = st.selectbox(
        "Select the GPT model version:",
        options=["gpt-3.5-turbo-1106", "gpt-4-1106-preview"],
        index=0  # Default to gpt-3.5-turbo
    )

    # Add reset button in sidebar
    if st.button('Reset Chat'):
        st.session_state.chat_history = []
        initialize_conversation()
        st.rerun()

if "context_hashes" not in st.session_state:
    st.session_state.context_hashes = set()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    initialize_conversation()

# Main chat container
chat_container = st.container()

# Handle chat input and display
with chat_container:
    user_input = st.chat_input("Ask something", key="user_input")

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"> **User**: {message['content']}")
        elif message["role"] == "model":
            st.markdown(f"> **Model**: {message['content']}")
        elif message["role"] == "context":
            with st.expander("Click to see the context"):
                for doc in message["content"]:
                    st.markdown(f"> **Context Document**: {doc.metadata['source']}")
                    st.markdown(f"> **Page Number**: {doc.metadata['page']}")
                    st.markdown(f"> **Content**: {doc.page_content}")


# Handle chat input and display
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    model_response, context = model_query(user_input, selected_files)
    st.session_state.chat_history.append({"role": "model", "content": model_response})
    if context:
        st.session_state.chat_history.append({"role": "context", "content": context})

    # Use st.rerun() to update the display immediately after sending the message
    st.rerun()