import os
 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
 
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
 
DB_FAISS_PATH = "vectorstore/db_faiss"
 
 
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db
 
 
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt
 
 
def load_medibot():
    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer user's question.
    If you dont know the answer, just say that you dont know.
    Dont provide anything out of the given context.
 
    Context: {context}
    Question: {question}
 
    Start the answer directly. No small talk please.
    """
 
    vectorstore = get_vectorstore()
 
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGroq(
            model = "llama-3.3-70b-versatile",
            temperature=0.0,
            groq_api_key=os.environ["GROQ_API_KEY"],
        ),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
        },
    )
 
    return qa_chain
 
 
def ask_medibot(chain, question):
    response = chain.invoke({"query": question})
    return response["result"]