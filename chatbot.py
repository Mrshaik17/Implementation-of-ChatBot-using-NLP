import nltk
import random
import os
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# SSL Fix for nltk downloads (Only if necessary)
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

# Ensure nltk_data folder exists
nltk_data_path = os.path.abspath('nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)
nltk.download('punkt')

# Define intents
intents = [
    {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey", "What's up", "How are you"],
     "responses": ["Hi there!", "Hello!", "Hey!", "Nothing much!", "I'm fine, thank you!"]},

    {"tag": "goodbye", "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
     "responses": ["Goodbye!", "See you later!", "Take care!"]},

    {"tag": "thanks", "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
     "responses": ["You're welcome!", "No problem!", "Glad I could help!"]},

    {"tag": "about", "patterns": ["What can you do?", "Who are you?", "What are you?", "What is your purpose?"],
     "responses": ["I am a chatbot!", "My purpose is to assist you.", "I can answer questions and provide assistance."]},

    {"tag": "help", "patterns": ["Help", "I need help", "Can you help me?", "What should I do?"],
     "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]}
]

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(stop_words="english")  # Removes common words
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []

for intent in intents:
    for pattern in intent["patterns"]:
        tags.append(intent["tag"])
        patterns.append(pattern.lower())  # Convert to lowercase for uniformity

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot function
def chatbot(input_text):
    input_text = input_text.lower()  # Convert user input to lowercase
    input_vector = vectorizer.transform([input_text])

    # Predict intent
    tag = clf.predict(input_vector)[0]

    # Find matching intent
    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "I'm sorry, I don't understand."

# Streamlit App Configuration
st.set_page_config(page_title="Chatbot", layout="wide")

# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "show_chat_history" not in st.session_state:
    st.session_state["show_chat_history"] = False  # Hide chat history by default

# Dummy user authentication (Replace with database logic)
USERS = {"Shaik": "test123", "Admin": "admin123"}

# Login Page
def login():
    st.title("🔒 Login to Chatbot")
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success("✅ Login successful! Redirecting...")
            st.rerun()
        else:
            st.error("❌ Invalid credentials, try again.")

# Dashboard Page (Chat Interface + Chat History in Sidebar)
def dashboard():
    st.sidebar.title("📌 Dashboard")
    st.sidebar.markdown(f"👋 Welcome, **{st.session_state['username']}**")

    # New Chat Button (Clears Chat History)
    if st.sidebar.button("🆕 New Chat"):
        st.session_state["chat_history"] = []
        st.success("Chat history cleared!")

    # Toggle Chat History Button
    if st.sidebar.button("📜 Toggle Chat History"):
        st.session_state["show_chat_history"] = not st.session_state["show_chat_history"]

    # Show Chat History if toggled
    if st.session_state["show_chat_history"]:
        st.sidebar.subheader("📜 Chat History")
        for message in reversed(st.session_state["chat_history"]):
            st.sidebar.text(message)

    # Styled welcome message
    st.markdown("<h1 style='text-align: center; color: white;'>🤖 Chatbot Assistant</h1>", unsafe_allow_html=True)
    st.write("💬 **Ask me anything below:**")

    # Get user input
    user_input = st.text_input("You: ", "", key="user_input", help="Type your message and press Enter")

    if user_input:
        response = chatbot(user_input)  # Call chatbot function
        st.session_state["chat_history"].append(f"👤 You: {user_input}")
        st.session_state["chat_history"].append(f"🤖 Chatbot: {response}")

        # Display chatbot response
        st.markdown(f"**🤖 Chatbot:** {response}")

    # Logout button
    if st.sidebar.button("🚪 Logout"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.session_state["chat_history"] = []
        st.rerun()  # Reloads the page

# Main Logic
if not st.session_state["logged_in"]:
    login()
else:
    dashboard()
