from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

import history_manager as hm

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

def get_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("vector_store", embeddings)
    retriever = db.as_retriever()
    llm = Ollama(model="llama3")  # Change to your model
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

qa_chain = get_chain()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    history = hm.load_history()
    return templates.TemplateResponse("chat.html", {"request": request, "history": history})

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    history = hm.load_history()
    hm.add_message(history, "user", user_input)

    response = qa_chain.run(user_input)
    hm.add_message(history, "bot", response)

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "history": history
    })
