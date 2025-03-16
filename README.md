# Implementation-of-ChatBot-using-NLP

Implementation of ChatBot using NLP:-
Introduction
This project is an intent-based chatbot built using Natural Language Processing (NLP) and Machine Learning (ML). The chatbot classifies user queries into predefined categories and provides instant responses using Logistic Regression and TfidfVectorizer. It features an interactive UI built with Streamlit, along with a secure authentication system, chat history, and a new chat option.

**Features:-**
✅ Predefined intent classification using NLP & ML

✅ Secure authentication system (Login & Logout)

✅ Interactive UI with Chat History and New Chat options

✅ Machine Learning-based intent detection using TfidfVectorizer and Logistic Regression

**Technologies Used:-**
Programming Language: Python 🐍

Frontend Framework: Streamlit 🎨

Machine Learning: Scikit-learn 🤖

Natural Language Processing (NLP): NLTK 📚

Vectorization: TfidfVectorizer 🔤

Model Used: Logistic Regression 📊

**Installation & Setup:-**
Follow these steps to run the chatbot on your local machine:

1️⃣ Clone the Repository 

2️⃣ Install Required Packages - "pip install -r requirements.txt"

3️⃣ Run the Chatbot - "python -m streamlit run chatbot.py"

**How It Works**
User inputs a message into the chatbot interface.

Text Processing: The message is tokenized and converted into numerical data using TfidfVectorizer.

Intent Classification: The Logistic Regression model predicts the user's intent.

Response Generation: The chatbot selects a predefined response based on the detected intent.

Chat History & User Authentication help in managing user sessions effectively.

**Limitations & Future Enhancements**
🚀 Current Limitations:
The chatbot only recognizes predefined intents and cannot generate responses dynamically.

**🔮 Future Enhancements:**

Integrating Large Language Models (LLMs) like GPT for dynamic AI-driven responses.

Adding voice-based interactions for better accessibility.

Improving accuracy by training on larger datasets.

Contributors
[A.S.Abdul.Touheed] - Developer


