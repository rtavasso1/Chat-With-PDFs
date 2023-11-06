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
import logging
logging.basicConfig(level=logging.INFO)

st.title("ChatGPT with Document Query")


# Define necessary embedding model, LLM, and vectorstore
openai_api_key = st.secrets["OPENAI_API_KEY"]
text_key = "text"

def hash_content(content):
    return hashlib.md5(content.encode('utf-8')).hexdigest()

@st.cache_resource()
def load_vectorstore():
    example_docs_path = Path("example-docs")
    filenames = list(example_docs_path.glob("*.pdf"))  # This will find all PDFs in the directory

    loaders = [PyPDFLoader(str(filename)) for filename in filenames]
    listOfPages = [loader.load_and_split() for loader in loaders]  # list of list of dict with keys "page_content", "metadata" {"source", "page"}
    
    faiss_indices = {str(filename): FAISS.from_documents(pages, OpenAIEmbeddings(disallowed_special=())) for filename, pages in zip(filenames, listOfPages)}  # dict of FAISS indexes
    
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
        for uploaded_file in uploaded_files:
            file_path = Path("example-docs") / uploaded_file.name
            with file_path.open("wb") as f:
                f.write(uploaded_file.getbuffer())
        faiss_indices = load_vectorstore()  # Reload the vectorstore with new files

    # Add a way to select which files to use for the model query
    selected_files = st.multiselect("Please select the files to query:", options=files)

    # Add a slider for number of sources to return 1-5
    num_sources = st.slider("Number of sources per document:", min_value=1, max_value=5, value=1)

    model_version = st.selectbox(
        "Select the GPT model version:",
        options=["gpt-3.5-turbo", "gpt-4"],
        index=0  # Default to gpt-3.5-turbo
    )

    # Add reset button in sidebar
    if st.button('Reset Chat'):
        st.session_state.chat_history = []
        initialize_conversation()
        st.experimental_rerun()

if "context_hashes" not in st.session_state:
    st.session_state.context_hashes = set()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    initialize_conversation()


# styl = f"""
# <style>
#     .stTextInput {{
#     position: fixed;
#     bottom: 3rem;
#     }}
#     .stButton {{
#     position: fixed;
#     bottom: 3rem;
#     right: 10rem;
#     }}
#     .stForm {{
#     position: fixed;
#     bottom: 3rem;
#     }}
# </style>
# """
# st.markdown(styl, unsafe_allow_html=True)

# Main chat container
chat_container = st.container()

# Handle chat input and display
with chat_container:
    # Then the chat input form at the bottom
    with st.form(key='user_input_form', clear_on_submit=True):
        user_input = st.text_input("Ask a question over the selected documents:", key="user_input")
        submit_button = st.form_submit_button(label='Submit')

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
if submit_button and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    model_response, context = model_query(user_input, selected_files)
    st.session_state.chat_history.append({"role": "model", "content": model_response})
    if context:
        st.session_state.chat_history.append({"role": "context", "content": context})
    # Clear the input box after the message is sent
    #st.session_state.user_input = ""

    # Use st.experimental_rerun() to update the display immediately after sending the message
    st.experimental_rerun()