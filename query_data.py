import streamlit as st
import random
import time
import os
from dotenv import load_dotenv
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env
load_dotenv()

# Get the OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Streamed response generator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def load_models():
    @st.cache(allow_output_mutation=True)
    def _load_models():
        embedding_function = OpenAIEmbeddings()
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function), ChatOpenAI()
    
    return _load_models()

def main():
    st.title("Chatbot with Streamlit")

    # Create chat interface
    st.write("## Chat Interface")

    # Create a container to hold the chat messages
    chat_container = st.container()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with chat_container:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Accept user input
    query_text = st.chat_input("You:")

    if query_text:
        db, model = load_models()

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        if len(results) == 0 or results[0][1] < 0.7:
            st.write("Bot: Unable to find matching results.")
            return

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        response_text = model.predict(prompt)

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Bot: {response_text}\nSources: {sources}"

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query_text})
        # Display user message
        with chat_container:
            with st.chat_message("user"):
                st.write(query_text)

        # Add bot's response to chat history
        st.session_state.messages.append({"role": "bot", "content": formatted_response})
        # Display bot's response
        with chat_container:
            with st.chat_message("bot"):
                st.write(formatted_response)

if __name__ == "__main__":
    main()
