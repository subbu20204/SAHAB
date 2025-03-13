from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOllama
from deep_translator import GoogleTranslator
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from config import GOOGLE_API_KEY

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

# Initialize session state
def initialize_session_state(st):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# Load and process your text data (Replace this with your actual legal text data)
def load_legal_data():
    text_data = """
    [Your legal text data here]
    """
    chunks = get_text_chunks(text_data)
    return get_vector_store(chunks)

# Function to get online response
def get_response_online(prompt, context):
    full_prompt = f"""
    As a legal chatbot specializing in the Indian Penal Code and Department of Justice services, you are tasked with providing highly accurate and contextually appropriate responses. Ensure your answers meet these criteria:
    - Respond in a bullet-point format to clearly delineate distinct aspects of the legal query or service information.
    - Each point should accurately reflect the breadth of the legal provision or service in question, avoiding over-specificity unless directly relevant to the user's query.
    - Clarify the general applicability of the legal rules, sections, or services mentioned, highlighting any common misconceptions or frequently misunderstood aspects.
    - Limit responses to essential information that directly addresses the user's question, providing concise yet comprehensive explanations.
    - When asked about live streaming of court cases, provide the relevant links for court live streams.
    - For queries about various DoJ services or information, provide accurate links and guidance.
    - Avoid assuming specific contexts or details not provided in the query, focusing on delivering universally applicable legal interpretations or service information unless otherwise specified.
    - Conclude with a brief summary that captures the essence of the legal discussion or service information and corrects any common misinterpretations related to the topic.

    CONTEXT: {context}
    QUESTION: {prompt}
    ANSWER:
    """
    return model.generate_content(full_prompt, stream=True)

# Function to get offline response
def get_response_offline(prompt, context):
    llm = ChatOllama(model="phi3")
    # Implement offline response generation here
    # This is a placeholder and needs to be implemented based on your offline requirements
    return "Offline mode is not fully implemented yet."

# Function to translate answer
def translate_answer(answer, target_language):
    translator = GoogleTranslator(source='auto', target=target_language)
    translated_answer = translator.translate(answer)
    return translated_answer

# Function to reset conversation
def reset_conversation(st):
    st.session_state.messages = []
    st.session_state.memory.clear()

# Function to get trimmed chat history
def get_trimmed_chat_history(st):
    max_history = 10
    return st.session_state.messages[-max_history:]

# Function to hide Streamlit's default menu
def hide_hamburger_menu(st):
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)
