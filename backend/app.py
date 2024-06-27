# backend/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain_community.callbacks.manager import get_openai_callback
import os
from dotenv import load_dotenv
import uvicorn
import uuid

load_dotenv()

os.environ["OPENAI_API_KEY"] = "..."

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

knowledge_bases = {}

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_reader = PdfReader(file.file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    # Generate a unique ID for this knowledge base
    knowledge_base_id = str(uuid.uuid4())
    knowledge_bases[knowledge_base_id] = knowledge_base
    
    return {"message": "PDF uploaded successfully", "knowledge_base_id": knowledge_base_id}


@app.post("/ask_question/")
async def ask_question(question: str = Form(...), knowledge_base_id: str = Form(...)):
    knowledge_base = knowledge_bases.get(knowledge_base_id)
    if not knowledge_base:
        return {"error": "Knowledge base not found"}
    
    docs = knowledge_base.similarity_search(question)
    
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=question)
    
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
