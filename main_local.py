import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()


if __name__ == "__main__":
    print("hi")
    pdf_path = "2210.03629v3.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(documents)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)
    print(docs)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")
    print("done")
    
    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings=embeddings, allow_dangerous_deserialization=True
    )
    print(new_vectorstore)
    
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(OpenAI(), retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=new_vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    res = retrival_chain.invoke(input={"input": "Give me the gist of ReAct in 3 sentences? 답변은 한글로 번역해서 말해줘"})
    print(res["answer"])
    
    
    
    