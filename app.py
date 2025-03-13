import time
import streamlit as st
from utils import (
    initialize_session_state, get_text_chunks, get_vector_store, load_legal_data,
    get_response_online, get_response_offline, translate_answer, reset_conversation,
    get_trimmed_chat_history, hide_hamburger_menu
)

# Set the Streamlit page configuration and theme
st.set_page_config(page_title="SAHAB", layout="wide")
st.header("SMART AUTOMATED HIERARCHICAL ANALYSIS BOT")

# Sidebar configuration
with st.sidebar:
    st.title("S.A.H.A.B")
    col1, col2, col3 = st.columns([1, 30, 1])
    with col2:
        st.image("images/Judge.png", use_column_width=True)
    model_mode = st.toggle("Online Mode", value=True)
    selected_language = st.selectbox("Start by Selecting your Language", 
                                     ["English", "Assamese", "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi", 
                                      "Nepali", "Odia", "Punjabi", "Sindhi", "Tamil", "Telugu", "Urdu"])

# Hide Streamlit's default menu
hide_hamburger_menu(st)

# Initialize session state
initialize_session_state(st)

# Load and process your text data
vector_store = load_legal_data()
db_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
input_prompt = st.chat_input("Start with your legal query")
if input_prompt:
    st.session_state.messages.append({"role": "user", "content": input_prompt})
    
    with st.chat_message("user"):
        st.markdown(input_prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = "⚠️ **_Gentle reminder: We generally ensure precise information, but do double-check._** \n\n"
        
        # Retrieve context
        context = db_retriever.get_relevant_documents(input_prompt)
        context_text = "\n".join([doc.page_content for doc in context])
        
        if model_mode:
            response_stream = get_response_online(input_prompt, context_text)
            for chunk in response_stream:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.01)  # Adjust for smoother effect
        else:
            response = get_response_offline(input_prompt, context_text)
            full_response += response
            message_placeholder.markdown(full_response)

        # Translate if necessary
        if selected_language != "English":
            translated_response = translate_answer(full_response, selected_language.lower())
            message_placeholder.markdown(translated_response)
        else:
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Add a reset button after each interaction
    if st.button('🗑️ Reset Conversation'):
        reset_conversation(st)
        st.experimental_rerun()

# Footer
def footer():
    st.markdown("""
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            color: black;
            text-align: center;
        }
        </style>
        <div class="footer">
        </div>
        """, unsafe_allow_html=True)

# Display the footer
footer()
