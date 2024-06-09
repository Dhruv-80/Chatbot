from flask import Flask, render_template, request
import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

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

app = Flask(__name__)

def load_models():
    def _load_models():
        embedding_function = OpenAIEmbeddings()
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function), ChatOpenAI()
    
    return _load_models()

@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        query_text = request.form["question"]
        db, model = load_models()

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        if len(results) == 0 or results[0][1] < 0.7:
            return render_template("index.html", response="Unable to find matching results.")

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        response_text = model.predict(prompt)

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        return render_template("index.html", response=formatted_response)
    return render_template("index.html", response=None)

if __name__ == "__main__":
    app.run(debug=True)
