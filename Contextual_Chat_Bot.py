# create python3 -m venv venv; mkdir db; mkdir pdf;
# run the shell script file with: chmod +x run_app.sh; ./run_app.sh

import os
import uuid
import time
import json
import sqlite3
import logging
from flask_cors import CORS
from datetime import datetime
from flask import Flask, request, g, jsonify
from langchain_community.llms import Ollama
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()

# Store chat history in memory
chat_history = []

UPLOAD_URLS = os.getenv("UPLOAD_URLS", "").split(",")  # Split into a list of URLs

# Create 'db' and 'pdf' directories if they don't exist
if not os.path.exists("db"):
    os.makedirs("db")

if not os.path.exists("pdf"):
    os.makedirs("pdf")

app = Flask(__name__)

# Allow requests from 'http://localhost:3000'
CORS(app)

# Set up logging for the main application log
LOG_FILE = "inference.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s %(message)s')

# Directory for chat log files
CHAT_LOG_DIR = "chat_logs"
if not os.path.exists(CHAT_LOG_DIR):
    os.makedirs(CHAT_LOG_DIR)

# Database setup
DB_PATH = os.path.join(app.root_path, 'chat.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS rag_chat_history_t (
        rag_id INTEGER PRIMARY KEY AUTOINCREMENT,
        chatbot_name VARCHAR(80),
        pid INTEGER,
        session_id TEXT,
        llm_name VARCHAR(40),
        input_prompt VARCHAR(240),
        prompt_time DATETIME,
        inference_result VARCHAR(640),
        inference_time DATETIME,
        duration_in_secs REAL
    )''')
    conn.commit()
    conn.close()

# Initialize the database
init_db()

# Helper function to log chat messages
def log_chat(message_type, content):
    with open(g.chat_log_filename, 'a') as f:
        f.write(f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Session ID: {g.session_id}\n")
        if message_type == "user":
            f.write(f"Question: {content}\n")
        else:
            f.write(f"Answer: {content}\n")
        f.write("\n")

@app.before_request
def before_request():
    if 'session_id' not in g:
        g.session_id = str(uuid.uuid4())
    if 'chat_log_filename' not in g:
        g.chat_log_filename = os.path.join(CHAT_LOG_DIR, f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{g.session_id}.log")

# Initialize LLM and embeddings
cached_llm = ChatOpenAI(model="gpt-4o", temperature=0.6) # gpt-3.5-turbo
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50, length_function=len, is_separator_regex=False)
raw_prompt = PromptTemplate.from_template(
    """
    Format your response in Markdown. Use appropriate headers, bullet points, code blocks, or tables as needed.
    Answer questions using the document, responding with "I don't know the answer" if no relevant answer is found.
    You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so.
    {input}
    Context: {context}
    Answer:
    """
)

def insert_chat_history(chatbot_name, pid, session_id, llm_name, input_prompt, prompt_time, inference_result, inference_time, duration_in_secs):
    logging.info(f"Inserting chat history: {chatbot_name}, {pid}, {session_id}, {llm_name}, {input_prompt}, {prompt_time}, {inference_result}, {inference_time}, {duration_in_secs}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''INSERT INTO rag_chat_history_t (chatbot_name, pid, session_id, llm_name, input_prompt, prompt_time, inference_result, inference_time, duration_in_secs)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', (chatbot_name, pid, session_id, llm_name, input_prompt, prompt_time, inference_result, inference_time, duration_in_secs))
    
    conn.commit()
    conn.close()

@app.route("/upload_documents", methods=["POST"])
def upload_documents():
    files = request.files.getlist("files")
    json_content = request.json if request.is_json else None
    request_urls = json_content.get("urls", []) if json_content else []

    # Load URLs from the environment variable
    env_urls = os.getenv("UPLOAD_URLS", "[]")  # Default to empty JSON array if not set
    try:
        env_urls = json.loads(env_urls).get("urls", [])
    except json.JSONDecodeError:
        return {"status": "Error", "message": "Invalid JSON format in UPLOAD_URLS environment variable."}

    # Combine request URLs with environment URLs
    all_urls = request_urls + env_urls
    all_docs = []

    # Handle file uploads
    for file in files:
        file_name = file.filename
        save_file = os.path.join("pdf", file_name)
        file.save(save_file)

        # Load documents based on file type
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(save_file)
            all_docs.extend(loader.load())
        elif file_name.endswith('.docx') or file_name.endswith('.doc'):
            loader = Docx2txtLoader(save_file)
            all_docs.extend(loader.load())
        elif file_name.endswith('.txt'):
            loader = TextLoader(save_file)
            all_docs.extend(loader.load())

    # Handle URLs input
    if all_urls:
        print(f"URLs provided: {all_urls}")
        try:
            loader = UnstructuredURLLoader(urls=all_urls)
            all_docs.extend(loader.load())
        except Exception as e:
            return {"status": "Error", "message": str(e)}

    if not all_docs:
        return {"status": "Error", "message": "No documents or URLs were processed."}

    chunks = text_splitter.split_documents(all_docs)

    if not chunks:
        return {"status": "Error", "message": "No chunks were created from the documents or URLs."}

    try:
        vector_store = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="db")
        vector_store.persist()
    except Exception as e:
        return {"status": "Error", "message": str(e)}

    response = {
        "status": "Successfully Uploaded",
        "filenames": [file.filename for file in files],
        "num_of_urls": len(all_urls),
        "urls": all_urls,  # Include combined URLs from request and environment
        "doc_len": len(all_docs),
        "chunks": len(chunks),
    }
    return jsonify(response)

def evaluate_answer(answer):
    generic_responses = ["I don't know", "I'm not sure", "I cannot answer that"]
    if any(phrase.lower() in answer.lower() for phrase in generic_responses) or not answer:
        return False  # Not useful
    return True  # Useful answer

@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")

    # Log the user's question
    log_chat("user", query)

    # Load vector store
    print("Loading vector store")
    vector_store = Chroma(persist_directory="db", embedding_function=embedding)

    # # Create retriever chain (Retrieve top 3 chunks based on similarity search)
    print("Creating retriever")
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Start timing
    start_time = time.time()

    # Perform retrieval of documents
    docs = retriever.get_relevant_documents(query)

    # Check if any documents were retrieved
    if not docs:
        answer = "I don't know the answer"  # Return a default answer if no documents are found
        sources = []
    else:
        # Prepare context for the prompt
        context = "\n".join([doc.page_content for doc in docs])  # Join page contents for context
        formatted_query = raw_prompt.format(input=query, context=context)  # Format the prompt

        # Process response using the chain
        response_chain = load_qa_chain(cached_llm, chain_type="stuff")
        result = response_chain.run(input_documents=docs, question=formatted_query)

        answer = result.strip()

        # Handle generic or empty answers
        generic_responses = ["I don't know", "I'm not sure", "I cannot answer that"]
        if any(phrase.lower() in answer.lower() for phrase in generic_responses) or not answer:
            answer = "Sorry, I couldn't find any relevant information based on the provided documents."

        sources = [{"source": doc.metadata["source"], "page_content": doc.page_content} for doc in docs]

    # End timing
    end_time = time.time()
    inference_time = end_time - start_time

    # Log the AI's answer
    log_chat("ai", answer)

    # Log the interaction into the database
    prompt_time = datetime.now()
    insert_chat_history(
        chatbot_name="Contextual_Chat_Bot",
        pid=os.getpid(),
        session_id=g.session_id,
        llm_name="gpt-3.5-turbo",
        input_prompt=query,
        prompt_time=prompt_time,
        inference_result=answer,
        inference_time=prompt_time,
        duration_in_secs=inference_time
    )
    
    useful_answer = evaluate_answer(answer)
    
    # Add to performance logs (can use DB or files)
    with open("performance_report.json", "a") as perf_log:
        json.dump({
            "query": query,
            "answer": answer,
            "useful": useful_answer,
            "inference_time": inference_time
        }, perf_log)
        perf_log.write("\n")

    # Prepare the response
    response_answer = {
        "answer": answer,
        "sources": sources,
        "inference_time": inference_time,
    }

    return jsonify(response_answer)

@app.route("/performance_report", methods=["GET"])
def performance_report():
    total_queries = 0
    useful_answers = 0
    total_inference_time = 0

    # Read performance logs
    with open("performance_report.json", "r") as perf_log:
        for line in perf_log:
            entry = json.loads(line)
            total_queries += 1
            if entry["useful"]:
                useful_answers += 1
            total_inference_time += entry["inference_time"]
    
    if total_queries > 0:
        useful_percentage = (useful_answers / total_queries) * 100
        avg_inference_time = total_inference_time / total_queries
    else:
        useful_percentage = 0
        avg_inference_time = 0

    report = {
        "total_queries": total_queries,
        "useful_answers": useful_answers,
        "useful_percentage": useful_percentage,
        "average_inference_time": avg_inference_time
    }

    return jsonify(report)


@app.route("/delete_vector_db", methods=["POST"])
def delete_vector_db():
    # Load the password from the environment variables
    correct_password = os.getenv("DELETE_DB_PASSWORD")
    
    # Retrieve the password provided by the user
    json_content = request.json
    provided_password = json_content.get("password")
    
    # Verify the password
    if not provided_password:
        return {"status": "Error", "message": "Password is required."}, 400
    
    if provided_password != correct_password:
        return {"status": "Error", "message": "Incorrect password."}, 403
    
    # Proceed to delete the vector database
    db_dir = os.path.join(app.root_path, "db")
    
    try:
        # Delete all files and folders in the db directory
        if os.path.exists(db_dir):
            for root, dirs, files in os.walk(db_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(db_dir)
        else:
            return {"status": "Error", "message": "Vector database not found."}, 404

        # Recreate the empty db folder
        os.makedirs(db_dir)
        
        return {"status": "Success", "message": "Vector database deleted and recreated successfully."}
    except Exception as e:
        return {"status": "Error", "message": str(e)}, 500

def start_app():
    app.run(host="0.0.0.0", port=8501, debug=True)

if __name__ == "__main__":
    start_app()
