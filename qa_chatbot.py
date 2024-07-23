import streamlit as st
import os
import base64
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from streamlit_chat import message
from constants import CHROMA_SETTINGS
import streamlit_scrollable_textbox as stx

st.set_page_config(page_title="Chatbot", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>

    .header {
        position: fixed;
        top: 6;
        width: 100%;
        background-color: #000000;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        text-align: center;
        color: #ffffff;
    }


    .main {
        background-color: #0d0d15;
        padding: 20px;
        flex: 1;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .chat-container {
        background-color: rgba(0, 0, 0, 0);
        top : 80;
        padding: 35px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        max-height: 500px;
        width: 100%;
        overflow-y: auto;
        padding-top = 60px;
    } border-radius: 5px;
    
    .block-container {
        padding : 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Fixed header
st.markdown('<div class="header"><h1 style="text-align: center;">⚕️ Health Insurance PDF Chat 📄</h1></div>', unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.header("Upload Your PDF Here")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

# Additional sidebar information
st.sidebar.markdown("<h4 style='color: grey;'>Built by Parth Sudan</h4>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='color: grey;'>MobileMSK LLC</h4>", unsafe_allow_html=True)

device = torch.device('cpu')
checkpoint = "C:\\Users\\GarvW\\Desktop\\Programs\\Chatbot\\Model6\\LaMini-T5-738M"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map=device,
    torch_dtype=torch.float32
)

persist_directory = "db"

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        "text2text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

pipeline = qa_llm()

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0, os.SEEK_SET)
    return size

def displayPDF(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="280" height="400" type="application/pdf"></iframe>'
    return pdf_display

def data_ingestion():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = text_splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()
    db = None

def process_answer(question):
    result = pipeline(question)
    return result["result"]

def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))

def main():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Hello! What would you like to know?"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]
    if "temp" not in st.session_state:
        st.session_state["temp"] = ""
    user_input = st.text_input("Your Question", key="input")

    if user_input:
        answer = process_answer(user_input)
        st.session_state["past"].append(user_input)
        response = answer
        st.session_state["generated"].append(response)
        # st.session_state["input"] = ""
    with st.container(height=400, border=True):
        if st.session_state["generated"]:
            display_conversation(st.session_state)

    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": get_file_size(uploaded_file)
        }
        filepath = os.path.join("docs", uploaded_file.name)
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        col1, col2 = st.sidebar.columns([1, 2])
        with col1:
            st.markdown("<div class = 'sidebar'><h4 style='color:black;'>File Details</h4>", unsafe_allow_html=True)
            st.json(file_details)
            st.markdown("<h4 style='color:black;'>File Preview</h4></div>", unsafe_allow_html=True)
            pdf_view = displayPDF(filepath)
            st.markdown(pdf_view, unsafe_allow_html=True)

        with col2:
            with st.spinner('The embeddings are being generated...'):
                data_ingestion()
            st.success('The embeddings have been created and uploaded successfully!')

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
