from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI

def run_llm_model(question: str = "") -> str:
    model_name = "gemma3"
    model = ChatOllama(model=model_name)
    chain = model | StrOutputParser()
    return chain.invoke(question)

if __name__ == "__main__":
    response = run_llm_model("What is the capital of France?")
    print(response) 