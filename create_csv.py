import csv
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os
import shutil
from dotenv import load_dotenv

CHROMA_PATH = "chroma"
DATA_PATH = "C:\\Users\\nadam_lwkruj7\\OneDrive\\Documents\\Code\\chatbot"

# Load environment variables from .env
load_dotenv()

# Get the OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.csv")  # Load CSV files
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = []
    for document in documents:
        # Assuming CSV structure: first column is the text content
        with open(document.file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                text = row[0]  # Assuming the text is in the first column
                chunks.extend(text_splitter.split(text))
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
