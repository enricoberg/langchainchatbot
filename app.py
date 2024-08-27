import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import argparse

#LOAD ALL THE CREDENTIALS FROM THE ENV FILE
load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
API_CLIENT=os.getenv("API_CLIENT")
API_SECRET=os.getenv("API_SECRET")
API_BASE_URL=os.getenv("API_BASE_URL")
open_api_headers={
  "client_id": API_CLIENT,
  "client_secret": API_SECRET
}
open_api_base=API_BASE_URL
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = open_api_base
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"



def get_pdf_text(pdf_docs):    
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    #CREATE EMBEDDINGS USING THE DIFFENTE CLASSES SPECIFIC TO GEMINI AND AZURE
    if "--gemini" in st.cli.argv:
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")
    else:
        embeddings=OpenAIEmbeddings(openai_api_base=API_BASE_URL,headers=open_api_headers, deployment="text-embedding-ada-002",model="text-embedding-ada-002", openai_api_key="not relevant", chunk_size=1, openai_api_type="azure", openai_api_version="2023-03-15-preview")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain(version):

    prompt_template = """    
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "Mi dispiace, non ho trovato una risposta nelle procedure  caricate a sistema...". don't provide the wrong answer. The answer MUST be in italian all the times.
    Always specifiy if the answer is not coming from the context, otherwise cite the source of the information\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    if version=="gemini":
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    else:
        model = AzureChatOpenAI(
        openai_api_base=open_api_base,
        openai_api_version="2023-03-15-preview",                      
        deployment_name="chatgpt4-32k",                                
        openai_api_key="not_relevant",
        headers = open_api_headers,
        max_tokens =8096,                                             
        temperature= 0.6)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain



def user_input(user_question,version):
    if version=="gemini":
        embeddings= GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")
    else:
        embeddings=OpenAIEmbeddings(openai_api_base=API_BASE_URL,headers=open_api_headers, deployment="text-embedding-ada-002",model="text-embedding-ada-002", openai_api_key="not relevant", chunk_size=1, openai_api_type="azure", openai_api_version="2023-03-15-preview")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(version)    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    #PRINT THE RESPONSE IN THE TEXTBOX
    st.write(":blue[B|Buddy: ]", response["output_text"])




def main(show_sidebar,version):
    
    
    
    st.set_page_config("Chat PDF")
    st.header("Chatta con B|Buddy 🤖")
    

    user_question = st.text_input("Chiedimi qualunque cosa, io conosco tutte le procedure BAI e sono qui per aiutarti:")

    if user_question:
        user_input(user_question,version)
    if show_sidebar:
        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Carica qui i tuoi file e premi Apprendi", accept_multiple_files=True)
            if st.button("Apprendi"):
                with st.spinner("Studiando..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Fatto")



if __name__ == "__main__":
    print("Show sidebar? y/n")
    show_sidebar = input()
    if show_sidebar == "y":
        show_sidebar = True
    else:
        show_sidebar = False
    print("Which model do you want to use? openai, gemini")
    version=input()
    if version == "gemini":
        version="gemini"
    else:
        version="openai"   
    print("Running...")
    main(show_sidebar,version)


    
