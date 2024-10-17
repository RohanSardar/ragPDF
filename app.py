import os
import tempfile
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title='PDF RAG', page_icon='⛓️', initial_sidebar_state='expanded')

if 'flag_uploaded' not in st.session_state:
        st.session_state.flag_uploaded = None
if 'chat_history' not in st.session_state:
        st.session_state.chat_history = ChatMessageHistory()

@st.cache_resource(show_spinner=False)
def setup_environment():
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "ragPDF"

    groq_api_key = st.secrets["GROQ_API_KEY"]
    os.environ['HF_TOKEN'] = st.secrets['HF_TOKEN']

    embeddings = HuggingFaceEmbeddings(model_name='Craig/paraphrase-MiniLM-L6-v2')

    st.sidebar.header('PDF')
    st.sidebar.write('Upload a PDF and chat with its content')

    st.title('🤖ChatBot with RAG on PDF📘')

    model = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-8b-8192')
    if 'flag_uploaded' not in st.session_state:
        st.session_state.flag_uploaded = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = ChatMessageHistory()

    return model, embeddings

with st.spinner('Configuring the environment'):
    model, embeddings = setup_environment()
if st.session_state.flag_uploaded == None:
    st.warning('Upload PDFs in the sidebar', icon='⬅️')

@st.cache_data(show_spinner=False)
def file_upload():
    if 'retriever' in st.session_state and 'vectorstore' in st.session_state:
        del st.session_state['retriever']
        del st.session_state['vectorstore']
    documents = []
    for pdf in pdfs:
        with st.sidebar.status('Reading PDFs'):
            pdf_bytes = pdf.getvalue() 
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf_bytes)
                temp_file_path = temp_file.name
            
            try:
                loader = PyPDFLoader(file_path=temp_file_path)
                docs = loader.load()
                documents.extend(docs)
              
            finally:
                os.remove(temp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    st.session_state.flag_uploaded = True

    if "retriever" not in st.session_state and st.session_state.flag_uploaded:
        with st.sidebar.status('Creating Vector DB'):
            st.session_state.vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            st.session_state.retriever = st.session_state.vectorstore.as_retriever()

pdfs = st.sidebar.file_uploader('Choose a PDF', type='pdf', 
                                accept_multiple_files=True, 
                                help='Upload a PDF file to ask query about it')
if st.sidebar.button('Submit', help='Click here to submit the PDFs', use_container_width=True):
    file_upload()

contextualize_q_system_prompt = (
    "Given a chat history and latest user question which "
    "might reference the context in the chat history, "
    "formulate an answer which is based on the context and latest question"
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', contextualize_q_system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}'),
    ]
)

if st.session_state.flag_uploaded:
    st.success('Now ask questions from PDF', icon='🙂')
    history_aware_retriever = create_history_aware_retriever(model, st.session_state.retriever, contextualize_q_prompt)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. Answer briefly"
        "\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}'),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history() -> BaseChatMessageHistory:
        return st.session_state.chat_history

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
    )

    session_history = get_session_history()

    user_input = st.chat_input('Ask here')
    if user_input:
        with st.toast('Thinking...', icon="🤔"):
            response = conversational_rag_chain.invoke({'input': user_input})
        for message in session_history.messages:
            if message.type == 'human':
                with st.chat_message('user'):
                    st.markdown(message.content)
            if message.type == 'ai':
                with st.chat_message('assistant'):
                    st.markdown(message.content)

footer_html = """<div style='text-align: center;'>
  <p style="font-size:80%; font-family: 'Trebuchet MS';">
  Developed by <a href="https://linktr.ee/RohanSardar">Rohan Sardar</a>
  <br>Project completed on 17th October 2024</p>
</div>"""
st.sidebar.markdown(footer_html, unsafe_allow_html=True)
