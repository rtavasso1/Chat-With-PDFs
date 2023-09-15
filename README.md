# Chat-With-PDFs

## Overview
Chat With PDFs is a conversational interface developed using Streamlit that provides document-based contextual information to ChatGPT. It combines GPT-based chat with FAISS to enable quick similarity search over documents. Users can ask questions, and the model provides answers using relevant sections of the selected documents as a context source.

## Features
- Query-based conversational context from multiple types of documents including `.pdf`, `.txt`, `.docx`, and `.xlsx`.
- Utilizes FAISS for efficient similarity search over documents.
- Allows the user to specify the number of sources per document for context.
- Allows for a resettable chat interface.
- Context is automatically added to the chat and can be viewed by expanding a dropdown.

## Usage
- Run the Streamlit app at tiny.utk.edu/EHulC
- Choose the pre-defined documents you want to query from.
- Specify the number of sources per document using the slider.
- Enter your query and hit 'Submit'.

## Resetting the Conversation
- You can reset the chat interface by clicking on the 'Reset Chat' button.

## License
This project is licensed under the MIT License.

## Acknowledgements
- OpenAI for the GPT model
- FAISS for efficient similarity search
- Streamlit for an intuitive web interface
