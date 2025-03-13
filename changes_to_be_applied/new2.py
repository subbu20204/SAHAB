import os
import time
import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOllama
from deep_translator import GoogleTranslator
from translate import Translator as OfflineTranslator
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langdetect import detect, DetectorFactory
import base64
from PIL import Image
import json
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static

DetectorFactory.seed = 0

# Import speech recognition and text-to-speech libraries
try:
    import speech_recognition as sr
    from gtts import gTTS
    from io import BytesIO
    import base64
    VOICE_FEATURES_AVAILABLE = True
except ImportError:
    VOICE_FEATURES_AVAILABLE = False

# Import Google API key from config
from config import GOOGLE_API_KEY

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

# Set the Streamlit page configuration and theme
st.set_page_config(page_title="", layout="wide")

# Define custom CSS for a more modern interface
def apply_custom_css():
    st.markdown("""
    <style>
        /* Main theme colors */
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #e74c3c;
            --background-color: #f9f9f9;
            --text-color: #333;
        }
        
        /* Body and background */
        body {
            background-color: var(--background-color);
            color: var(--text-color);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Header styling */
        h1, h2, h3 {
            color: var(--primary-color);
            font-weight: 600;
        }
        
        /* Main header with animation */
        .main-header {
            background: linear-gradient(45deg, #2c3e50, #3498db);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            animation: gradient 5s ease infinite;
            background-size: 200% 200%;
        }
        
        @keyframes gradient {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        
        /* Chat message styling */
        .user-message {
            background-color: #e3f2fd;
            border-left: 5px solid #2196f3;
            padding: 10px 15px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        .assistant-message {
            background-color: #f1f8e9;
            border-left: 5px solid #689f38;
            padding: 10px 15px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        /* Button styling */
        .stButton button {
            background-color: var(--secondary-color);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            background-color: var(--primary-color);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transform: translateY(-2px);
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #f5f5f5;
            border-right: 1px solid #ddd;
        }
        
        /* Feature cards */
        .feature-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            margin-bottom: 15px;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        /* Voice button */
        .voice-button {
            background-color: var(--accent-color);
            color: white;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto;
            font-size: 24px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .voice-button:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        }
        
        /* Progress bars */
        .stProgress > div > div {
            background-color: var(--secondary-color);
        }
        
        /* Hide hamburger menu */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Animated typing indicator */
        .typing-indicator {
            display: inline-block;
            position: relative;
        }
        
        .typing-indicator::after {
            content: '';
            position: absolute;
            width: 6px;
            height: 15px;
            background-color: var(--text-color);
            right: -12px;
            top: 2px;
            animation: blink 0.7s infinite;
        }
        
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0; }
        }
        
        /* Language selector styling */
        .language-selector {
            background-color: white;
            border-radius: 8px;
            padding: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }
        
        /* Input box styling */
        .chat-input {
            border-radius: 20px;
            border: 2px solid #e0e0e0;
            padding: 10px 15px;
            margin-top: 10px;
            transition: all 0.3s ease;
        }
        
        .chat-input:focus {
            border-color: var(--secondary-color);
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.3);
        }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# Function to translate text dynamically
def translate_text(text, target_language):
    try:
        # Attempt online translation
        translated_text = GoogleTranslator(source='auto', target=target_language.lower()).translate(text)
        return translated_text
    except Exception as e:
        st.warning("Online translation failed, attempting offline translation.")
        try:
            # Attempt offline translation
            offline_translator = OfflineTranslator(to_lang=target_language.lower())
            translated_text = offline_translator.translate(text)
            return translated_text
        except Exception as e:
            st.error(f"Translation failed: {str(e)}")
            return text  # Return original text if translation fails

# Sidebar configuration with improved styling
with st.sidebar:
    st.title("R.A.Y.S")
    col1, col2, col3 = st.columns([1, 30, 1])

    # Language selection
    selected_language = st.selectbox(
        "Select your preferred language",
        ["English", "Assamese", "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi", "Nepali", "Odia", "Punjabi", "Sindhi", "Tamil", "Telugu", "Urdu"]
    )

    # Translate sidebar content
    st.markdown(f"""
    <div class="feature-card">
        <h3>{translate_text("🌟 About RAYS", selected_language)}</h3>
        <p>{translate_text("RAYS is designed to make legal assistance accessible to all, especially in rural and underserved communities. Get accurate legal information in your preferred language through text or voice interaction.", selected_language)}</p>
    </div>
    """, unsafe_allow_html=True)

    model_mode = st.toggle(translate_text("Online Mode", selected_language), value=True)

    # Voice features toggle
    if VOICE_FEATURES_AVAILABLE:
        voice_enabled = st.toggle(translate_text("Enable Voice Interaction", selected_language), value=True)
        if voice_enabled:
            st.markdown(f'<p>{translate_text("🎙️ Voice interaction is enabled. Click the microphone button to speak your query.", selected_language)}</p>', unsafe_allow_html=True)
    else:
        voice_enabled = False
        st.warning(translate_text("Voice libraries not installed. Run: pip install SpeechRecognition gtts", selected_language))

    # Legal resources section
    st.markdown(f"""
    <div class="feature-card">
        <h3>{translate_text("📚 Legal Resources", selected_language)}</h3>
        <ul>
            <li><a href="https://www.legalservicesindia.com/">{translate_text("Legal Services India", selected_language)}</a></li>
            <li><a href="https://nalsa.gov.in/">{translate_text("National Legal Services Authority", selected_language)}</a></li>
            <li><a href="https://doj.gov.in/">{translate_text("Department of Justice", selected_language)}</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Emergency contacts
    st.markdown(f"""
    <div class="feature-card">
        <h3>{translate_text("🆘 Emergency Contacts", selected_language)}</h3>
        <p><strong>{translate_text("National Legal Aid Helpline:", selected_language)}</strong> 15100</p>
        <p><strong>{translate_text("Women Helpline:", selected_language)}</strong> 1091</p>
        <p><strong>{translate_text("Child Helpline:", selected_language)}</strong> 1098</p>
    </div>
    """, unsafe_allow_html=True)

# Main content
st.markdown(f'<div class="main-header"><h1>{translate_text("Rights Assistance for Youth and Society", selected_language)}</h1><p>{translate_text("Your AI-powered legal assistant for accessible justice", selected_language)}</p></div>', unsafe_allow_html=True)

# Tabs for different functionalities
tabs = st.tabs([
    translate_text("💬 Chat Assistant", selected_language),
    translate_text("🗺️ Legal Aid Locator", selected_language),
    translate_text("📊 Legal Awareness", selected_language),
    translate_text("❓ FAQ", selected_language),
    translate_text("📄 Document Verification", selected_language)
])
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

def text_to_speech(text, lang_code="en"):
    language_codes = {
        "English": "en",
        "Hindi": "hi",
        "Tamil": "ta",
        "Telugu": "te",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Bengali": "bn",
        "Gujarati": "gu",
        "Marathi": "mr",
        "Punjabi": "pa",
        "Urdu": "ur",
        "Assamese": "as",
        "Odia": "or",
        "Nepali": "ne",
        "Sindhi": "sd"
    }
    
    try:
        tts = gTTS(text=text, lang=language_codes.get(selected_language, "en"), slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        audio_base64 = base64.b64encode(fp.read()).decode()
        audio_html = f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")

# Function to listen to voice input
def listen_for_voice():
    if not VOICE_FEATURES_AVAILABLE:
        return None

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.markdown("<p>Listening... Please speak your query.</p>", unsafe_allow_html=True)
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5)
        
    try:
        language_codes = {
            "English": "en-US",
            "Hindi": "hi-IN",
            "Tamil": "ta-IN",
            "Telugu": "te-IN",
            "Kannada": "kn-IN", 
            "Malayalam": "ml-IN",
            "Bengali": "bn-IN",
            "Gujarati": "gu-IN",
            "Marathi": "mr-IN", 
            "Punjabi": "pa-IN",
            "Urdu": "ur-IN"
        }
        
        lang_code = language_codes.get(selected_language, "en-US")
        text = recognizer.recognize_google(audio, language=lang_code)
        return text
    except sr.UnknownValueError:
        st.warning("Sorry, I couldn't understand what you said.")
    except sr.RequestError:
        st.error("Could not request results; check your network connection")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    return None
# Load and process your text data (Replace this with your actual legal text data)
text_data = """
[Your legal text data here]
"""

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_text(text_data)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")
vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

# Convert vector store into a retriever
db_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Now you can use db_retriever in your code

with tabs[0]:  # Chat Assistant Tab
    # Create two columns for chat and voice input
    chat_col, voice_col = st.columns([5, 1])

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><strong>RAYS:</strong> {message["content"]}</div>', unsafe_allow_html=True)

    
    
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
        - When providing legal information, also mention if free legal aid may be available for the situation.
        - If asked about legal aid centers, mention that users can check the Legal Aid Locator tab.

        CONTEXT: {context}
        QUESTION: {prompt}
        ANSWER:
        """
        response = model.generate_content(full_prompt, stream=True)
        return response

    def get_response_offline(prompt, context):
        llm = ChatOllama(model="phi3")
        # Implement offline response generation here
        # This is a placeholder and needs to be implemented based on your offline requirements
        return "Offline mode is not fully implemented yet."

  
    def translate_answer(answer, target_language):
        try:
        # Attempt online translation
          translated_answer = GoogleTranslator(source='auto', target=target_language.lower()).translate(answer)
          return translated_answer
        except Exception as e:
           st.warning("Online translation failed, attempting offline translation.")
        try:
            # Attempt offline translation
            offline_translator = OfflineTranslator(to_lang=target_language.lower())
            translated_answer = offline_translator.translate(answer)
            return translated_answer
        except Exception as e:
            st.error(f"Offline translation failed: {str(e)}")
            return answer 

    def reset_conversation():
        st.session_state.messages = []
        st.session_state.memory.clear()

    def get_trimmed_chat_history():
        max_history = 10
        return st.session_state.messages[-max_history:]

    # Display messages with improved styling
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><strong>RAYS:</strong> {message["content"]}</div>', unsafe_allow_html=True)

    # Voice input button in the voice column
    with voice_col:
        if VOICE_FEATURES_AVAILABLE and voice_enabled:
            if st.button("🎤", help="Click to speak your query"):
                with st.spinner("Listening..."):
                    voice_input = listen_for_voice()
                    if voice_input:
                        st.session_state.voice_input = voice_input

    # Handle user input (either from text or voice)
    input_prompt = None
    
    # Check if there's voice input in session state
    if hasattr(st.session_state, 'voice_input') and st.session_state.voice_input:
        input_prompt = st.session_state.voice_input
        st.session_state.voice_input = None  # Clear after use
    else:
        # Regular text input
        input_prompt = st.chat_input("Start with your legal query", key="chat_input")
    
    if input_prompt:
        st.markdown(f'<div class="user-message"><strong>You:</strong> {input_prompt}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": input_prompt})
        trimmed_history = get_trimmed_chat_history()

        with st.spinner("Thinking 💡..."):
            context = db_retriever.get_relevant_documents(input_prompt)
            context_text = "\n".join([doc.page_content for doc in context])
            
            if model_mode:
                response = get_response_online(input_prompt, context_text)
            else:
                response = get_response_offline(input_prompt, context_text)

            message_placeholder = st.empty()
            full_response = "⚠️ *Gentle reminder: We generally ensure precise information, but do double-check.* \n\n\n"
            
            if model_mode:
                for chunk in response:
                    full_response += chunk.text
                    time.sleep(0.02)  # Adjust the sleep time to control the "typing" speed
                    message_placeholder.markdown(f'<div class="assistant-message"><strong>RAYS:</strong> {full_response}</div>', unsafe_allow_html=True)
            else:
                full_response += response
                message_placeholder.markdown(f'<div class="assistant-message"><strong>RAYS:</strong> {full_response}</div>', unsafe_allow_html=True)

            # Translate the answer to the selected language
            if selected_language != "English":
                with st.spinner(f"Translating to {selected_language}..."):
                    translated_answer = translate_answer(full_response, selected_language.lower())
                    message_placeholder.markdown(f'<div class="assistant-message"><strong>RAYS:</strong> {translated_answer}</div>', unsafe_allow_html=True)
                    
                    # Play TTS for the translated response if voice is enabled
                    if VOICE_FEATURES_AVAILABLE and voice_enabled:
                        text_to_speech(translated_answer, selected_language.lower())
            else:
                # Play TTS for English response if voice is enabled
                if VOICE_FEATURES_AVAILABLE and voice_enabled:
                    text_to_speech(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Reset button
    if st.button('🗑️ Reset Conversation', on_click=reset_conversation):
        st.rerun()

with tabs[1]:  # Legal Aid Locator Tab
    st.markdown(f"""
     <h2>{translate_text("🗺️ Find Legal Aid Centers Near You", 
     selected_language)}</h2>
        <p>{translate_text("Enter your location to find legal aid centers and free legal services available in your area.",
         selected_language)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    location_input = st.text_input(translate_text("Enter your city or area", selected_language), "Delhi")
    if st.button(translate_text("Find Legal Aid Centers", selected_language)):
           with st.spinner(translate_text("Locating centers...", selected_language)):
            user_location = get_user_location(location_input)
            if user_location:
                user_lat, user_lon = user_location
                legal_map = create_legal_aid_map(user_lat, user_lon)
                st.markdown("<h3>Legal Aid Centers Near You</h3>", unsafe_allow_html=True)
                folium_static(legal_map)
                
                st.markdown(f"""
                <div class="feature-card">
                    <h3>{translate_text("Free Legal Aid Eligibility", selected_language)}</h3>
                    <p>{translate_text("In India, free legal services are available to:", selected_language)}</p>
                    <ul>
                        <li>{translate_text("Women and children", selected_language)}</li>
                        <li>{translate_text("Victims of trafficking", selected_language)}</li>
                        <li>{translate_text("Persons with disabilities", selected_language)}</li>
                        <li>{translate_text("Victims of mass disaster, ethnic violence, caste atrocity, flood, drought, earthquake, industrial disaster", selected_language)}</li>
                        <li>{translate_text("Industrial workmen", selected_language)}</li>
                        <li>{translate_text("Persons in custody", selected_language)}</li>
                        <li>{translate_text("Persons with annual income less than Rs. 1,00,000 (may vary by state)", selected_language)}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(translate_text("Location not found. Please try a different city or check your spelling.", selected_language))
with tabs[2]:  # Legal Awareness Tab
    st.markdown(f"""
    <div class="feature-card">
        <h2>{translate_text("📊 Legal Awareness", selected_language)}</h2>
        <p>{translate_text("Educational resources to understand your legal rights and responsibilities.", selected_language)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Translate legal topics
    legal_topics = [
        translate_text("Women's Rights", selected_language),
        translate_text("Property Laws", selected_language),
        translate_text("Consumer Protection", selected_language),
        translate_text("Labor Laws", selected_language),
        translate_text("Right to Information", selected_language)
    ]
    
    # Translate the selectbox label
    selected_topic = st.selectbox(translate_text("Select a topic to learn about", selected_language), legal_topics)
    
    # Translate topic content dynamically
    topic_content = {
        translate_text("Women's Rights", selected_language): f"""
        <div class="feature-card">
            <h3>{translate_text("📝 Women's Legal Rights in India", selected_language)}</h3>
            <ul>
                <li><strong>{translate_text("Protection from Domestic Violence:", selected_language)}</strong> {translate_text("Under the Protection of Women from Domestic Violence Act, 2005.", selected_language)}</li>
                <li><strong>{translate_text("Equal Pay:", selected_language)}</strong> {translate_text("The Equal Remuneration Act, 1976 mandates equal pay for equal work.", selected_language)}</li>
                <li><strong>{translate_text("Maternity Benefits:", selected_language)}</strong> {translate_text("The Maternity Benefit Act provides for 26 weeks of paid maternity leave.", selected_language)}</li>
                <li><strong>{translate_text("Protection at Workplace:", selected_language)}</strong> {translate_text("Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013.", selected_language)}</li>
                <li><strong>{translate_text("Property Rights:", selected_language)}</strong> {translate_text("Equal inheritance rights under the Hindu Succession (Amendment) Act, 2005.", selected_language)}</li>
            </ul>
            <p><strong>{translate_text("Helplines:", selected_language)}</strong> {translate_text("Women's Helpline: 1091, Domestic Abuse Helpline: 181", selected_language)}</p>
        </div>
        """,
        
        translate_text("Property Laws", selected_language): f"""
        <div class="feature-card">
            <h3>{translate_text("📝 Property Laws - Key Points", selected_language)}</h3>
            <ul>
                <li><strong>{translate_text("Registration:", selected_language)}</strong> {translate_text("All property transactions should be registered under the Registration Act, 1908.", selected_language)}</li>
                <li><strong>{translate_text("Stamp Duty:", selected_language)}</strong> {translate_text("Mandatory payment varying by state (typically 5-10% of property value).", selected_language)}</li>
                <li><strong>{translate_text("Inheritance:", selected_language)}</strong> {translate_text("Governed by personal laws (Hindu, Muslim, Christian, Parsi) or Indian Succession Act.", selected_language)}</li>
                <li><strong>{translate_text("Tenant Rights:", selected_language)}</strong> {translate_text("Protected under various Rent Control Acts in different states.", selected_language)}</li>
                <li><strong>{translate_text("Land Ceiling:", selected_language)}</strong> {translate_text("Restrictions on maximum land holdings in urban areas.", selected_language)}</li>
            </ul>
        </div>
        """,
        
        translate_text("Consumer Protection", selected_language): f"""
        <div class="feature-card">
            <h3>{translate_text("📝 Consumer Protection Rights", selected_language)}</h3>
            <p>{translate_text("Under the Consumer Protection Act, 2019, you have the right to:", selected_language)}</p>
            <ul>
                <li><strong>{translate_text("Right to Safety:", selected_language)}</strong> {translate_text("Protection against hazardous goods and services.", selected_language)}</li>
                <li><strong>{translate_text("Right to Information:", selected_language)}</strong> {translate_text("Complete details about performance, quality, quantity, and price.", selected_language)}</li>
                <li><strong>{translate_text("Right to Choose:", selected_language)}</strong> {translate_text("Access to variety of goods at competitive prices.", selected_language)}</li>
                <li><strong>{translate_text("Right to be Heard:", selected_language)}</strong> {translate_text("Have your interests receive due consideration.", selected_language)}</li>
                <li><strong>{translate_text("Right to Redressal:", selected_language)}</strong> {translate_text("Fair settlement of genuine grievances.", selected_language)}</li>
                <li><strong>{translate_text("Right to Consumer Education:", selected_language)}</strong> {translate_text("Acquire knowledge and skills to be an informed consumer.", selected_language)}</li>
            </ul>
            <p><strong>{translate_text("How to File a Complaint:", selected_language)}</strong> {translate_text("Visit the nearest Consumer Forum or file online at", selected_language)} <a href="https://consumerhelpline.gov.in">{translate_text("National Consumer Helpline", selected_language)}</a></p>
        </div>
        """,
        
        translate_text("Labor Laws", selected_language): f"""
        <div class="feature-card">
            <h3>{translate_text("📝 Key Labor Laws in India", selected_language)}</h3>
            <ul>
                <li><strong>{translate_text("Minimum Wages Act, 1948:", selected_language)}</strong> {translate_text("Ensures minimum wage payment to workers.", selected_language)}</li>
                <li><strong>{translate_text("Factories Act, 1948:", selected_language)}</strong> {translate_text("Regulates working conditions in factories.", selected_language)}</li>
                <li><strong>{translate_text("Payment of Gratuity Act, 1972:", selected_language)}</strong> {translate_text("Provides for gratuity payment to employees.", selected_language)}</li>
                <li><strong>{translate_text("Employees' Provident Fund Act:", selected_language)}</strong> {translate_text("Ensures retirement benefits.", selected_language)}</li>
                <li><strong>{translate_text("Payment of Bonus Act, 1965:", selected_language)}</strong> {translate_text("Provides for annual bonus payment.", selected_language)}</li>
                <li><strong>{translate_text("Industrial Disputes Act, 1947:", selected_language)}</strong> {translate_text("Mechanism for settlement of industrial disputes.", selected_language)}</li>
            </ul>
            <p><strong>{translate_text("For Labor Disputes:", selected_language)}</strong> {translate_text("Contact your local Labor Commissioner Office", selected_language)}</p>
        </div>
        """,
        
        translate_text("Right to Information", selected_language): f"""
        <div class="feature-card">
            <h3>{translate_text("📝 Right to Information (RTI) Act, 2005", selected_language)}</h3>
            <p><strong>{translate_text("What is RTI?", selected_language)}</strong> {translate_text("A law that allows citizens to request information from any public authority.", selected_language)}</p>
            <ul>
                <li><strong>{translate_text("How to File an RTI:", selected_language)}</strong> {translate_text("Submit application with Rs. 10 fee to the Public Information Officer (PIO).", selected_language)}</li>
                <li><strong>{translate_text("Time Limit:", selected_language)}</strong> {translate_text("Information must be provided within 30 days (48 hours if life/liberty is involved).", selected_language)}</li>
                <li><strong>{translate_text("Appeal Process:", selected_language)}</strong> {translate_text("First appeal to designated officer, second appeal to Information Commission.", selected_language)}</li>
                <li><strong>{translate_text("Exemptions:", selected_language)}</strong> {translate_text("Information affecting sovereignty, security, strategic interests, trade secrets, privacy, etc.", selected_language)}</li>
            </ul>
            <p><strong>{translate_text("Online RTI Filing:", selected_language)}</strong> <a href="https://rtionline.gov.in">{translate_text("RTI Online Portal", selected_language)}</a></p>
        </div>
        """
    }
    
    # Display the translated topic content
    st.markdown(topic_content[selected_topic], unsafe_allow_html=True)
    
    # Translate video resources section
    st.markdown(f"""
    <div class="feature-card">
        <h3>{translate_text("📺 Educational Videos", selected_language)}</h3>
        <p>{translate_text("Watch informative videos to better understand legal concepts", selected_language)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    video_col1, video_col2 = st.columns(2)
    with video_col1:
        st.markdown(f"#### {translate_text('Know Your Rights', selected_language)}")
        st.image("https://via.placeholder.com/400x225", caption=translate_text("Legal awareness video", selected_language))
    
    with video_col2:
        st.markdown(f"#### {translate_text('How to File a Police Complaint', selected_language)}")
        st.image("https://via.placeholder.com/400x225", caption=translate_text("Process overview video", selected_language))

with tabs[3]:  # FAQ Tab
    st.markdown(f"""
    <div class="feature-card">
        <h2>{translate_text("❓ Frequently Asked Questions", selected_language)}</h2>
        <p>{translate_text("Find answers to common legal questions and concerns.", selected_language)}</p>
    </div>
    """, unsafe_allow_html=True)

    # FAQ Accordion
    with st.expander(translate_text("1. How can I get free legal aid in India?", selected_language)):
        st.markdown(f"""
        <div class="feature-card">
            <p>{translate_text("Free legal aid is available through the", selected_language)} <strong>{translate_text("National Legal Services Authority (NALSA)", selected_language)}</strong> {translate_text("and its state-level counterparts. You can:", selected_language)}</p>
            <ul>
                <li>{translate_text("Visit your nearest", selected_language)} <strong>{translate_text("District Legal Services Authority (DLSA)", selected_language)}</strong>.</li>
                <li>{translate_text("Call the NALSA helpline at", selected_language)} <strong>15100</strong>.</li>
                <li>{translate_text("Apply online through the", selected_language)} <a href="https://nalsa.gov.in">{translate_text("NALSA website", selected_language)}</a>.</li>
            </ul>
            <p><strong>{translate_text("Eligibility:", selected_language)}</strong> {translate_text("Women, children, SC/ST communities, victims of trafficking, and individuals with an annual income below ₹1,00,000 are eligible.", selected_language)}</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander(translate_text("2. What should I do if I'm a victim of domestic violence?", selected_language)):
        st.markdown(f"""
        <div class="feature-card">
            <p>{translate_text("If you're facing domestic violence, take the following steps:", selected_language)}</p>
            <ul>
                <li>{translate_text("Call the", selected_language)} <strong>{translate_text("Women's Helpline", selected_language)}</strong> {translate_text("at", selected_language)} <strong>1091</strong> {translate_text("or", selected_language)} <strong>181</strong>.</li>
                <li>{translate_text("File a complaint under the", selected_language)} <strong>{translate_text("Protection of Women from Domestic Violence Act, 2005", selected_language)}</strong>.</li>
                <li>{translate_text("Contact a", selected_language)} <strong>{translate_text("Protection Officer", selected_language)}</strong> {translate_text("in your district.", selected_language)}</li>
                <li>{translate_text("Seek help from NGOs like", selected_language)} <strong>{translate_text("Majlis", selected_language)}</strong> {translate_text("or", selected_language)} <strong>{translate_text("SAKHI", selected_language)}</strong>.</li>
            </ul>
            <p><strong>{translate_text("Note:", selected_language)}</strong> {translate_text("You can also approach the nearest police station or family court for immediate assistance.", selected_language)}</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander(translate_text("3. How do I file a consumer complaint?", selected_language)):
        st.markdown(f"""
        <div class="feature-card">
            <p>{translate_text("To file a consumer complaint:", selected_language)}</p>
            <ul>
                <li>{translate_text("Approach the", selected_language)} <strong>{translate_text("Consumer Forum", selected_language)}</strong> {translate_text("in your district.", selected_language)}</li>
                <li>{translate_text("File a complaint online at the", selected_language)} <a href="https://consumerhelpline.gov.in">{translate_text("National Consumer Helpline", selected_language)}</a>.</li>
                <li>{translate_text("Provide evidence such as bills, receipts, and correspondence.", selected_language)}</li>
            </ul>
            <p><strong>{translate_text("Time Limit:", selected_language)}</strong> {translate_text("Complaints must be filed within 2 years of the issue.", selected_language)}</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander(translate_text("4. What are my rights as a tenant?", selected_language)):
        st.markdown(f"""
        <div class="feature-card">
            <p>{translate_text("As a tenant, you have the following rights:", selected_language)}</p>
            <ul>
                <li><strong>{translate_text("Right to a Rent Agreement:", selected_language)}</strong> {translate_text("Ensure you have a written agreement.", selected_language)}</li>
                <li><strong>{translate_text("Protection from Eviction:", selected_language)}</strong> {translate_text("Landlords cannot evict you without proper notice.", selected_language)}</li>
                <li><strong>{translate_text("Right to Essential Services:", selected_language)}</strong> {translate_text("Landlords must provide water, electricity, and maintenance.", selected_language)}</li>
                <li><strong>{translate_text("Security Deposit:", selected_language)}</strong> {translate_text("You are entitled to the return of your deposit upon vacating.", selected_language)}</li>
            </ul>
            <p><strong>{translate_text("Note:", selected_language)}</strong> {translate_text("Tenant rights vary by state. Check your state's Rent Control Act for specifics.", selected_language)}</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander(translate_text("5. How do I file an RTI application?", selected_language)):
        st.markdown(f"""
        <div class="feature-card">
            <p>{translate_text("To file an RTI application:", selected_language)}</p>
            <ul>
                <li>{translate_text("Write a clear application stating the information you need.", selected_language)}</li>
                <li>{translate_text("Pay a fee of ₹10 (waived for below-poverty-line applicants).", selected_language)}</li>
                <li>{translate_text("Submit the application to the", selected_language)} <strong>{translate_text("Public Information Officer (PIO)", selected_language)}</strong> {translate_text("of the relevant department.", selected_language)}</li>
                <li>{translate_text("You can file online at the", selected_language)} <a href="https://rtionline.gov.in">{translate_text("RTI Online Portal", selected_language)}</a>.</li>
            </ul>
            <p><strong>{translate_text("Response Time:", selected_language)}</strong> {translate_text("Information must be provided within 30 days.", selected_language)}</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander(translate_text("6. What should I do if I'm arrested?", selected_language)):
        st.markdown(f"""
        <div class="feature-card">
            <p>{translate_text("If you're arrested, remember the following:", selected_language)}</p>
            <ul>
                <li><strong>{translate_text("Right to Know:", selected_language)}</strong> {translate_text("You have the right to know the reason for your arrest.", selected_language)}</li>
                <li><strong>{translate_text("Right to Legal Aid:", selected_language)}</strong> {translate_text("You can request free legal aid from the nearest Legal Services Authority.", selected_language)}</li>
                <li><strong>{translate_text("Right to Bail:", selected_language)}</strong> {translate_text("For bailable offenses, you can apply for bail immediately.", selected_language)}</li>
                <li><strong>{translate_text("Right to Inform:", selected_language)}</strong> {translate_text("The police must inform a family member or friend about your arrest.", selected_language)}</li>
            </ul>
            <p><strong>{translate_text("Note:", selected_language)}</strong> {translate_text("Do not sign any documents without consulting a lawyer.", selected_language)}</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander(translate_text("7. How can I check the status of my court case?", selected_language)):
        st.markdown(f"""
        <div class="feature-card">
            <p>{translate_text("To check the status of your court case:", selected_language)}</p>
            <ul>
                <li>{translate_text("Visit the", selected_language)} <a href="https://ecourts.gov.in">{translate_text("eCourts website", selected_language)}</a>.</li>
                <li>{translate_text("Enter your", selected_language)} <strong>{translate_text("CNR (Case Number Record)", selected_language)}</strong>.</li>
                <li>{translate_text("You can also visit the court's website or contact the court clerk.", selected_language)}</li>
            </ul>
            <p><strong>{translate_text("Note:", selected_language)}</strong> {translate_text("You can also use the", selected_language)} <strong>{translate_text("Legal Aid Locator", selected_language)}</strong> {translate_text("tab to find nearby courts.", selected_language)}</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander(translate_text("8. What are my rights as a woman in the workplace?", selected_language)):
        st.markdown(f"""
        <div class="feature-card">
            <p>{translate_text("As a woman in the workplace, you have the following rights:", selected_language)}</p>
            <ul>
                <li><strong>{translate_text("Equal Pay:", selected_language)}</strong> {translate_text("You are entitled to equal pay for equal work under the Equal Remuneration Act, 1976.", selected_language)}</li>
                <li><strong>{translate_text("Maternity Benefits:", selected_language)}</strong> {translate_text("You are entitled to 26 weeks of paid maternity leave under the Maternity Benefit Act.", selected_language)}</li>
                <li><strong>{translate_text("Protection from Harassment:", selected_language)}</strong> {translate_text("The Sexual Harassment of Women at Workplace Act, 2013 protects you from harassment.", selected_language)}</li>
                <li><strong>{translate_text("Safe Working Conditions:", selected_language)}</strong> {translate_text("Employers must provide a safe and harassment-free workplace.", selected_language)}</li>
            </ul>
            <p><strong>{translate_text("Note:", selected_language)}</strong> {translate_text("If you face any issues, report them to your HR department or the Internal Complaints Committee (ICC).", selected_language)}</p>
        </div>
        """, unsafe_allow_html=True)

with tabs[4]:  # Document Verification Tab
    st.markdown(f"""
    <div class="feature-card">
        <h2>{translate_text("📄 Document Verification", selected_language)}</h2>
        <p>{translate_text("Upload your document to verify its authenticity. You can upload from your local file system or cloud storage.", selected_language)}</p>
    </div>
    """, unsafe_allow_html=True)

    # Create columns for file upload options
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {translate_text('Upload from File Explorer', selected_language)}")
        uploaded_file = st.file_uploader(translate_text("Choose a file", selected_language), type=["pdf", "docx", "txt"], key="file_uploader")

    with col2:
        st.markdown(f"### {translate_text('Upload from Cloud Storage', selected_language)}")
        cloud_option = st.selectbox(translate_text("Select Cloud Storage", selected_language), ["Google Drive", "Dropbox", "OneDrive"])
        if cloud_option == "Google Drive":
            st.write(translate_text("Google Drive integration coming soon!", selected_language))
        elif cloud_option == "Dropbox":
            st.write(translate_text("Dropbox integration coming soon!", selected_language))
        elif cloud_option == "OneDrive":
            st.write(translate_text("OneDrive integration coming soon!", selected_language))

    # Document verification logic
    if uploaded_file is not None:
        st.markdown(f"### {translate_text('Uploaded Document Details', selected_language)}")
        file_details = {
            translate_text("Filename", selected_language): uploaded_file.name,
            translate_text("File Type", selected_language): uploaded_file.type,
            translate_text("File Size", selected_language): f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.json(file_details)

        # Simulate verification logic (for demonstration purposes)
        def verify_document(file):
            # Example: Check if the file is a PDF and has a reasonable size
            if file.type == "application/pdf" and file.size < 5 * 1024 * 1024:  # Less than 5MB
                return True, translate_text("Document is verified and authentic.", selected_language)
            else:
                return False, translate_text("Document verification failed. Please check the file format and size.", selected_language)

        # Perform verification
        is_verified, verification_message = verify_document(uploaded_file)

        # Display verification result
        if is_verified:
            st.success(f"✅ {translate_text('Verified', selected_language)}")
            st.markdown(f"<div class='assistant-message'>{verification_message}</div>", unsafe_allow_html=True)
        else:
            st.error(f"❌ {translate_text('Not Verified', selected_language)}")
            st.markdown(f"<div class='assistant-message'>{verification_message}</div>", unsafe_allow_html=True)

        # Display the uploaded file (if PDF)
        if uploaded_file.type == "application/pdf":
            st.markdown(f"### {translate_text('Document Preview', selected_language)}")
            base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" style="border: 1px solid #ddd;"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style="background-color: #2c3e50; padding: 20px; border-radius: 10px; text-align: center; margin-top: 30px; color: white;">
    <p>{translate_text("RAYS- Empowering Citizens with Legal Knowledge", selected_language)}</p>
    <p style="font-size: 0.8em;">{translate_text("This is an AI-powered legal assistant. For specific legal advice, consult a qualified legal professional.", selected_language)}</p>
    <p style="font-size: 0.8em;">{translate_text("Emergency Contacts: National Legal Aid Helpline: 15100 | Women's Helpline: 1091 | Child Helpline: 1098", selected_language)}</p>
</div>
""", unsafe_allow_html=True)