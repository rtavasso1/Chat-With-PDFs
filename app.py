import streamlit as st
import os
from langchain.vectorstores import FAISS
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import pickle

st.title("ChatGPT with Document Query")

# Define necessary embedding model, LLM, and vectorstore

text_key = "text"
with open('faiss_indices.pkl', 'rb') as f:
    faiss_indices = pickle.load(f)


def initialize_conversation():
    chat = ChatOpenAI(temperature=0)
    template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

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


# Read the files from the directory
directory = r"./example-docs"
files = []
if directory:
    for file in os.listdir(directory):
        if file.endswith(".txt") or file.endswith(".pdf") or file.endswith(".docx") or file.endswith(".xlsx"):
            #if file == 'imagebind.pdf': file = 'ImageBind.pdf'
            #if file == 'megabyte.pdf': file = 'MegaByte.pdf'
            files.append('example-docs/'+file)
st.write("Uploads are unavailable. The available files are listed below.")

# Add a way to select which files to use for the model query
selected_files = st.multiselect("Please select the files to query:", options=files)

# Add a slider for number of sources to return 1-5
num_sources = st.slider("Number of sources per document:", min_value=1, max_value=5, value=1)

def getTopK(query, doc_name):
    related = faiss_indices[doc_name].similarity_search(query, k=num_sources)
    return related

def model_query(query, document_names):
    
    if len(st.session_state.chat_history) == 1:
        all_related = []
        for document_name in document_names:
            related = getTopK(query, document_name)
            all_related.extend(related)
        all_related = sorted(all_related, key=lambda x: x[1], reverse=True)

        related = all_related[:num_sources]
        related = [r[0] for r in related] # remove scores
        context = " ".join([r.page_content for r in related])
        ai_message = st.session_state.conversation.predict(input=context+query)
        return ai_message, related
    else:
        ai_message = st.session_state.conversation.predict(input=query)
        return ai_message, None
    #response = f"Sample response for the query '{query}' over the documents: {', '.join(document_names)}"
    

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    initialize_conversation()

if st.button('Reset Chat'):
    st.session_state.chat_history = []
    initialize_conversation()
    st.experimental_rerun()

#user_input = st.text_input("Ask a question over the selected documents:")
with st.form(key='user_input_form'):
    user_input = st.text_input("Ask a question over the selected documents:")
    submit_button = st.form_submit_button(label='Submit')


if submit_button:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    model_response, context = model_query(user_input, selected_files)
    st.session_state.chat_history.append({"role": "model", "content": model_response})
    if context:
        st.session_state.chat_history.append({"role": "context", "content": context})
    user_input = ""

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

