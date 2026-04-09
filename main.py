from fastapi import FastAPI
from pydantic import BaseModel
from medibot import load_medibot, ask_medibot
 
app = FastAPI(
    title="MediBot API",
    docs_url="/docs",
    redoc_url="/redoc"
)
 
 
# Load once at startup
rag_chain = load_medibot()
 
 
class Query(BaseModel):
    question: str
 
 
@app.get("/")
def root():
    return {"message": "MediBot API running"}
 
 
@app.post("/chat")
def chat(query: Query):
    answer = ask_medibot(rag_chain, query.question)
    return {"response": answer}