import json, os
from fastapi import FastAPI
from langserve import add_routes
from langchain_ollama import OllamaLLM as Ollama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load configuration from key.json
with open("key.json", "r") as f:
    config = json.load(f)

# Setting up environment variables for Langsmith and Groq
# Commented out to avoid tracking right now, uncomment if needed

# os.environ["LANGCHAIN_API_KEY"] = config["LANGCHAIN_API_KEY"]
# os.environ["LANGCHAIN_TRACING_V2"] = config["LANGCHAIN_TRACING_V2"]
# os.environ["LANGCHAIN_PROJECT"] = config["LANGCHAIN_PROJECT_NAME"]
# os.environ["GROQ_API_KEY"] = config["GROQ_API_KEY"]


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that translates English to {language}."),
        ("user", "Translate the following: '{text}'"),
    ]
)

if config["llm_service"] == "paid":
    # Initialize the Groq Chat model : Paid inference
    llm = ChatGroq(model="Gemma2-9b-It", api_key=os.environ["GROQ_API_KEY"])
    model_name = "Gemma2-9b-It - GROQ"
else:
    # Ollama model initialization
    llm = Ollama(model="llama3.2")
    model_name = "Llama 3.2 - Ollama"

chain = prompt | llm | StrOutputParser()

app = FastAPI(
    title="Language Translation API - " + model_name,
    description="API for translating text between languages",
    version="1.0"
)
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

add_routes(app, chain, path="/chain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)