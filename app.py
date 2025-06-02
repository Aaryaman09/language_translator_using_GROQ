import json, os
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
import streamlit as st

# Load configuration from key.json
with open("key.json", "r") as f:
    config = json.load(f)

# Ensure the Ollama server is running and accessible on Langsmith server
os.environ["LANGCHAIN_API_KEY"] = config["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = config["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_PROJECT"] = config["LANGCHAIN_PROJECT_NAME"]
os.environ["GROQ_API_KEY"] = config["GROQ_API_KEY"]

if config["llm_service"] == "paid":
    # Initialize the Groq Chat model : Paid inference
    llm = ChatGroq(model="Gemma2-9b-It", api_key=os.environ["GROQ_API_KEY"])
    model_name = "Gemma2-9b-It - GROQ"
else:
    # Ollama model initialization
    llm = Ollama(model="llama3.2")
    model_name = "Llama 3.2 - Ollama"

# Creating more robust prompt with ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that translates English to {language}."),
        ("user", "Translate the following English text to {language}: '{text}'"),
    ]
)

# Create a chain with the prompt and the LLM
chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    # Create a Streamlit app to interact with the model
    st.title(f"Language Translation with {model_name}")
    st.write("Enter the language and text to translate:")
    language = st.text_input("Language (e.g., Hindi, French):", "Hindi")
    text = st.text_input("Text to translate:", "Hello, how are you?")

    if st.button("Translate"):
        if not language or not text:
            st.error("Please provide both language and text.")
        else:
            # Invoke the model with user input
            model_response = chain.invoke({"language": language, "text": text})

            # Display the model response in the Streamlit app
            st.title("Translation Result")
            st.write(model_response)
