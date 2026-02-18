import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

# ===== LOAD VECTORSTORE =====
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

# ===== CUSTOM MEDICAL PROMPT =====
CUSTOM_PROMPT_TEMPLATE = """
You are a medical information assistant.

Use ONLY the provided context to answer the question.
Do NOT diagnose or prescribe medicine.
If the answer is not in the context, say "I don't know".

Always include this disclaimer at the end:
"⚠️ This information is for educational purposes only. Please consult a qualified healthcare professional."

Context:
{context}

Question:
{question}

Answer:
"""

def get_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

# ===== STREAMLIT UI =====
def main():
    st.set_page_config(page_title="AI Health Assistant", page_icon="🩺")
    st.title("🩺 HealthBot")
    

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_query = st.chat_input("Ask health-related questions")

    if user_query:
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        try:
            vectorstore = load_vectorstore()

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                    temperature=0.0,
                    groq_api_key=os.environ["GROQ_API_KEY"],
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=False,   # 🔥 NO SOURCE DOCS SHOWN
                chain_type_kwargs={"prompt": get_prompt()}
            )

            response = qa_chain.invoke({"query": user_query})
            answer = response["result"]

            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
