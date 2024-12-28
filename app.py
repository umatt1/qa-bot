import streamlit as st
from qa_bot import get_answer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Insurance QA Bot",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
    }
    .stTextArea > div > div > textarea {
        background-color: #f0f2f6;
    }
    .main {
        padding: 2rem;
    }
    .stMarkdown {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🤖 Insurance QA Assistant")
st.markdown("""
Ask me anything about car insurance! I'm trained on Allstate's knowledge base and can help you understand:
- Coverage types and limits
- Policy features and benefits
- Insurance terms and concepts
- Claims processes
""")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your insurance question..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_answer(prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
    This QA bot uses advanced AI to answer your insurance questions using Allstate's official resources.
    
    **Features:**
    - Real-time answers
    - Source citations
    - Up-to-date information
    
    **Data Source:**
    Information comes from Allstate's official resources and documentation.
    
    **Note:**
    This is a demo application. For official insurance advice, please consult with an Allstate agent.
    """)
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()
