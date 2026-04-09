import streamlit as st
import requests
 
API_URL = "http://127.0.0.1:8000/chat"
 
st.title("MediBot")
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])
 
prompt = st.chat_input("Ask a medical question...")
 
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
 
    response = requests.post(
        API_URL,
        json={"question": prompt}
    )
 
    answer = response.json()["response"]
 
    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})