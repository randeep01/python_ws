from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

import history_manager as hm

import os

app = FastAPI()

# Setup
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global QA chain
qa_chain = None


def get_chain():
    global qa_chain
    if qa_chain is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.load_local("vector_store", embeddings)
        retriever = db.as_retriever()
        llm = Ollama(model="llama3")  # You can change the model here
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    history = hm.load_history()
    return templates.TemplateResponse("chat.html", {"request": request, "history": history})


@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    qa = get_chain()
    history = hm.load_history()
    hm.add_message(history, "user", user_input)

    response = qa.run(user_input)
    hm.add_message(history, "bot", response)

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "history": history
    })
