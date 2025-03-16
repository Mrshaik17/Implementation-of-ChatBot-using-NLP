#Login Details:-
#username:- Admin
#password:- admin123


import nltk
import random
import os
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datetime import datetime

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

# Define intents with more categories
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
     "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]},

    {"tag": "weather", "patterns": ["What's the weather like?", "Tell me the weather", "How's the weather?", "Weather update"],
     "responses": ["I can help with that! Can you tell me your location?", "I'm not sure about the weather, but I can look it up for you if you'd like."]},

    {"tag": "joke", "patterns": ["Tell me a joke", "Make me laugh", "I need a joke", "Tell a funny joke"],
     "responses": ["Why don't skeletons fight each other? They don't have the guts.", "Why was the math book sad? It had too many problems.", "What do you call fake spaghetti? An impasta."]},

    {"tag": "quote", "patterns": ["Give me a quote", "Tell me a quote", "Inspirational quote", "Motivational quote"],
     "responses": ["The only way to do great work is to love what you do. – Steve Jobs", "Success is not the key to happiness. Happiness is the key to success. – Albert Schweitzer", "Believe you can and you're halfway there. – Theodore Roosevelt"]},

    {"tag": "news", "patterns": ["What's the news?", "Tell me the latest news", "Any updates?", "What's happening today?"],
     "responses": ["I don't have real-time news, but you can check your favorite news website!", "For the latest news, you can visit a news portal like BBC or CNN."]},

    {"tag": "sports", "patterns": ["What's the score?", "How's the game going?", "Tell me the sports scores", "Any sports updates?"],
     "responses": ["I don't have live sports updates, but you can visit ESPN for the latest scores.", "Check your favorite sports app for real-time scores!"]},

    {"tag": "love", "patterns": ["I love you", "You're amazing", "I feel great", "I love talking to you"],
     "responses": ["Aw, that's so sweet of you! 😊", "You're awesome too! 😄", "Thank you! I'm here to help!"]},

    {"tag": "advice", "patterns": ["I need advice", "Can you give me advice?", "I don't know what to do", "What should I do?"],
     "responses": ["When in doubt, take a step back and breathe. You'll figure it out.", "Don't be afraid to ask for help when you need it.", "Trust yourself; you've got this!"]},

    {"tag": "name", "patterns": ["What's your name?", "Who are you?", "What should I call you?", "What are you called?"],
     "responses": ["I am a chatbot, you can call me whatever you'd like!", "You can call me Chatbot, I'm here to help!"]},

    {"tag": "music", "patterns": ["Play some music", "I need music", "Can you play a song?", "What should I listen to?"],
     "responses": ["I can't play music, but I can suggest some playlists! How about a chill playlist on Spotify?", "For some great music, check out Apple Music or YouTube!"]},

    {"tag": "movie", "patterns": ["Tell me a movie suggestion", "What movie should I watch?", "Give me a movie recommendation", "What should I watch today?"],
     "responses": ["How about a classic? 'The Shawshank Redemption' is always a great choice!", "You can check out 'Inception' or 'The Matrix' for a mind-bending experience!", "I recommend 'The Godfather' or 'The Dark Knight' if you're in the mood for some great action."]},

    {"tag": "shopping", "patterns": ["I want to shop", "Where can I buy this?", "Can you help me with shopping?", "Where can I get this product?"],
     "responses": ["You can check out Amazon or eBay for a variety of products!", "I recommend visiting your favorite online shopping website like Best Buy or Target!"]},

    {"tag": "motivation", "patterns": ["I need motivation", "Can you motivate me?", "Give me some motivation", "Inspire me"],
     "responses": ["Keep pushing, you're doing great! Every small step counts.", "Believe in yourself and great things will happen!", "Success is the sum of small efforts, repeated day in and day out."]},

    {"tag": "error", "patterns": ["Something went wrong", "I have an error", "There's a problem", "I need help with an error"],
     "responses": ["I'm sorry about that! Can you provide more details so I can help?", "Oops, something went wrong. Please try again.", "Could you please explain the error a bit more? I'll try to help."]},

    {"tag": "feedback", "patterns": ["I have feedback", "Here's my feedback", "Can I provide feedback?", "I want to leave feedback"],
     "responses": ["We value your feedback! Please share your thoughts.", "Thank you for your feedback! We are always working to improve.", "We appreciate your input! Let us know how we can improve."]},

    {"tag": "location", "patterns": ["Where are you located?", "Where is your office?", "Where can I find you?", "What is your location?"],
     "responses": ["I exist in the digital world, so I don't have a physical location!", "I don't have a physical office. I'm here online!"]},

    {"tag": "timezone", "patterns": ["What time is it?", "What is your time zone?", "What's the time right now?", "Can you tell me the time?"],
     "responses": ["I don't have a specific time zone. However, I can help you figure out the time in your location!", "Time is an abstract concept for me, but I can check the time in your location!"]},

    {"tag": "birthday", "patterns": ["When is your birthday?", "Do you have a birthday?", "What is your birthday?"],
     "responses": ["I don't have a birthday since I'm a program!", "As a digital assistant, I don't celebrate birthdays, but I'd be happy to help you plan one!"]},

    {"tag": "holiday", "patterns": ["What are the upcoming holidays?", "When is the next holiday?", "Tell me about upcoming holidays", "What holidays are coming up?"],
     "responses": ["That depends on where you're located! Could you tell me your country?", "You can check local calendars for upcoming holidays!"]},

    {"tag": "recipes", "patterns": ["Can you give me a recipe?", "What should I cook?", "Do you have a recipe for dinner?", "Give me a recipe for a meal"],
     "responses": ["How about a simple pasta recipe? Boil pasta, sauté garlic and olive oil, then mix! Or let me know what ingredients you have and I can help with ideas.", "For a quick dinner, you can make a stir-fry with your favorite veggies and protein."]},

    {"tag": "quotes", "patterns": ["Tell me a motivational quote", "Give me a famous quote", "I need a quote for today", "Share a quote with me"],
     "responses": ["“The best way to predict the future is to create it.” – Abraham Lincoln", "“You miss 100% of the shots you don't take.” – Wayne Gretzky", "“Success is not final, failure is not fatal: It is the courage to continue that counts.” – Winston Churchill"]},

    {"tag": "exercise", "patterns": ["Can you suggest an exercise?", "What should I do for a workout?", "I need exercise tips", "What exercise can I do today?"],
     "responses": ["You can try some squats and push-ups for a simple full-body workout!", "How about a quick run or some yoga stretches to start your day?", "You can try a 10-minute HIIT workout to get your heart pumping!"]},

    {"tag": "fitness", "patterns": ["How do I get fit?", "Can you give me fitness advice?", "I want to be fit", "Help me with fitness"],
     "responses": ["Stay consistent with your workouts and maintain a balanced diet. Always listen to your body!", "Start with small, achievable goals and increase intensity over time. Keep moving and challenge yourself!", "Mix up your routine with cardio, strength, and flexibility exercises!"]},

    {"tag": "travel", "patterns": ["Where should I travel?", "What are some travel destinations?", "Can you recommend travel places?", "Tell me about popular tourist spots"],
     "responses": ["How about visiting Paris for some culture, Rome for history, or Bali for some relaxation?", "If you love nature, you could visit Iceland or the Swiss Alps!", "Tokyo and New York are great for city lovers, while the Maldives is perfect for a beach getaway!"]},

    {"tag": "fitness_goal", "patterns": ["How do I set a fitness goal?", "What should my fitness goal be?", "Can you help me set fitness goals?", "I need help with my fitness plan"],
     "responses": ["Start with specific and achievable goals like running 5K in a month or lifting a certain weight. Consistency is key!", "Set SMART goals: Specific, Measurable, Achievable, Relevant, and Time-bound.", "You can aim for something like a fitness challenge, weight loss, or a strength training target."]},

    {"tag": "sleep", "patterns": ["How can I sleep better?", "I need sleep tips", "Help me fall asleep", "Tell me how to improve my sleep"],
     "responses": ["Try to keep a consistent sleep schedule and avoid caffeine in the afternoon.", "Meditation or reading before bed can help you relax.", "Make sure your bedroom is quiet, cool, and dark for better sleep quality!"]},

    {"tag": "study", "patterns": ["Can you help me study?", "I need study tips", "How can I study better?", "Give me advice on studying"],
     "responses": ["Try using the Pomodoro technique: study for 25 minutes and take a 5-minute break.", "Make sure to take breaks and stay hydrated. Find a quiet space for your study.", "Break your study materials into chunks and prioritize topics you struggle with the most."]},

    {"tag": "mental_health", "patterns": ["How do I take care of my mental health?", "I need mental health advice", "Help me with mental health", "Can you give me tips for mental well-being?"],
     "responses": ["Try practicing mindfulness or meditation to clear your mind.", "Taking regular breaks, exercising, and speaking to a professional can help with mental well-being.", "It's important to talk to someone you trust, take time for yourself, and stay active."]},

    {"tag": "technology", "patterns": ["Tell me about technology", "What is the latest in tech?", "What are the new tech trends?", "Give me technology news"],
     "responses": ["AI and machine learning are huge trends right now. Have you heard about OpenAI?", "The rise of 5G, autonomous vehicles, and blockchain are revolutionizing the tech industry.", "Wearable tech, augmented reality, and virtual reality are also growing rapidly."]},

    {"tag": "career", "patterns": ["What career should I choose?", "I need help with my career", "What career options do I have?", "Give me career advice"],
     "responses": ["Think about what you're passionate about and how it aligns with your skills and values.", "Consider growing in areas like tech, healthcare, or finance, as they are in demand.", "Networking, skill development, and internships can be great ways to discover and advance your career."]},

    {"tag": "self_improvement", "patterns": ["How can I improve myself?", "Give me self-improvement tips", "I want to grow as a person", "How can I better myself?"],
     "responses": ["Start with setting small, realistic goals and track your progress.", "Reading books, learning new skills, and reflecting on your daily habits can all improve your personal growth.", "Stay consistent and patient with your journey toward self-improvement."]},

    {"tag": "finances", "patterns": ["Can you help me with finances?", "I need financial advice", "Tell me about managing money", "How can I manage my finances?"],
     "responses": ["Start by budgeting and saving a portion of your income for emergencies.", "Consider investing in stocks, bonds, or real estate once you understand the basics of finance.", "Educate yourself about financial literacy. There's plenty of free resources online!"]},

    {"tag": "shopping", "patterns": ["Can I shop online?", "What are the best shopping websites?", "Where can I buy this?", "Tell me about shopping platforms"],
     "responses": ["You can shop on websites like Amazon, eBay, or Walmart for most products.", "For fashion, try stores like ASOS, Zara, or H&M.", "Check out Etsy for unique handmade items!"]},

    {"tag": "environment", "patterns": ["What can I do for the environment?", "Give me tips on being eco-friendly", "How can I help the environment?", "What are some environmental tips?"],
     "responses": ["Consider reducing, reusing, and recycling to minimize waste.", "Opt for renewable energy sources like solar power. Also, driving less can help reduce your carbon footprint.", "Support sustainable brands and reduce plastic use whenever possible."]},

    {"tag": "languages", "patterns": ["What languages do you speak?", "Can you help me with a language?", "Do you know other languages?", "Can I learn a language from you?"],
     "responses": ["I can communicate in English, but I can help you learn phrases in other languages too!", "I can translate common phrases for you in multiple languages. Just ask!"]},

    {"tag": "life", "patterns": ["What is the meaning of life?", "Give me a deep life quote", "What should I do with my life?", "Tell me about life"],
     "responses": ["The meaning of life is something different for everyone. What brings you joy and fulfillment?", "Life is all about finding purpose, loving, learning, and growing.", "A deep quote: 'The purpose of life is not to be happy. It is to be useful, to be honorable, to be compassionate, to have it make some difference that you have lived and lived well.' – Ralph Waldo Emerson."]},
    
    {"tag": "travel_tips", "patterns": ["Can you give me some travel tips?", "What should I know before traveling?", "Any advice for first-time travelers?", "How do I prepare for a trip?"],
    "responses": ["Always check the weather forecast before packing.", "Make sure to have a travel insurance for safety.", "Keep digital and physical copies of important documents like passport and tickets."]},

    {"tag": "best_places", "patterns": ["Where are the best places to visit?", "Recommend top travel destinations", "What are the best cities to travel to?", "Where should I go for a vacation?"],
    "responses": ["Top destinations include Paris, Tokyo, New York, and Barcelona.", "For adventure, try hiking in the Swiss Alps or visiting the Grand Canyon!", "Consider exploring Southeast Asia with its beautiful beaches and rich culture."]},

    {"tag": "nutrition", "patterns": ["What should I eat to stay healthy?", "Can you recommend a healthy diet?", "Give me nutrition tips", "What food is good for energy?"],
    "responses": ["Include plenty of fruits and vegetables in your diet.", "Try a balanced diet with proteins, carbs, and healthy fats.", "Stay hydrated and limit processed foods for better health."]},

    {"tag": "workout_routines", "patterns": ["What is a good workout routine?", "Can you suggest an exercise plan?", "Give me some fitness exercises", "What is a good workout to stay in shape?"],
     "responses": ["You can try a mix of cardio, strength training, and flexibility exercises.", "A simple routine could be running for 20 minutes, followed by 20 minutes of weightlifting.", "Try 3-4 days of weight training, along with 2 days of cardio for a balanced workout."]},

    {"tag": "stress_management", "patterns": ["How can I manage stress?", "Give me tips for reducing stress", "What should I do if I'm feeling stressed?", "Help me relieve stress"],
    "responses": ["Try deep breathing exercises or meditation to calm your mind.", "Take breaks throughout the day and spend time outdoors to reduce stress.", "A good sleep routine and physical exercise also help reduce stress."]},

    {"tag": "depression", "patterns": ["How do I deal with depression?", "What can I do when I feel depressed?", "Give me advice for overcoming sadness", "How can I lift my mood?"],
    "responses": ["It's important to talk to a professional if you're feeling depressed.", "Try journaling to express your thoughts and feelings.", "Regular exercise, a healthy diet, and quality sleep are key to improving your mood."]},

    {"tag": "motivation", "patterns": ["I need motivation", "Give me a motivational quote", "I'm feeling down, can you motivate me?", "Can you inspire me today?"],
    "responses": ["“The only way to do great work is to love what you do.” – Steve Jobs", "Remember: 'Success is not final, failure is not fatal: It is the courage to continue that counts.' – Winston Churchill", "Every small step counts towards your bigger goal. Keep going!"]},

    {"tag": "productivity", "patterns": ["How can I be more productive?", "Give me tips to be more productive", "I feel unproductive, help me out", "How do I improve my productivity?"],
    "responses": ["Try using the Pomodoro technique – 25 minutes of focused work followed by a 5-minute break.", "Prioritize tasks and break them into smaller chunks to make them more manageable.", "Eliminate distractions by turning off notifications and setting specific time blocks for work."]},

    {"tag": "relationships", "patterns": ["How do I improve my relationship?", "Give me relationship advice", "What should I do to keep my relationship strong?", "How can I be a better partner?"],
    "responses": ["Communication is key to any strong relationship. Be honest and listen actively.", "Make time for each other and plan regular date nights or activities.", "Show appreciation and support each other’s goals and passions."]},

    {"tag": "dating", "patterns": ["How do I approach dating?", "Give me dating advice", "What should I know about dating?", "How do I meet new people?"],
     "responses": ["Be yourself and don't rush things. Take time to get to know the person.", "It's important to communicate openly and have fun while dating.", "Be respectful and consider each other's feelings and boundaries."]},

    {"tag": "tech_trends", "patterns": ["What are the latest technology trends?", "Can you tell me about the latest tech?", "What’s new in technology?", "Tell me about current tech innovations?"],
    "responses": ["Artificial Intelligence and Machine Learning are transforming industries.", "5G, Blockchain, and the Internet of Things (IoT) are leading the tech revolution.", "Virtual Reality (VR) and Augmented Reality (AR) are also booming in gaming and education."]},

    {"tag": "gadgets", "patterns": ["What’s the latest in gadgets?", "Can you recommend some gadgets?", "What gadgets should I buy?", "Tell me about cool tech gadgets"],
    "responses": ["Smartphones, tablets, and smartwatches are always evolving. The latest iPhone and Samsung Galaxy are excellent choices.", "Look into smart home devices like Alexa, Google Nest, and smart thermostats.", "Wireless earbuds and noise-canceling headphones are popular choices for tech enthusiasts."]},

    {"tag": "budgeting", "patterns": ["How do I create a budget?", "Give me tips on budgeting", "How should I manage my finances?", "Help me create a budget plan"],
    "responses": ["Start by tracking your income and expenses. Categorize them into necessities and luxuries.", "Set savings goals and review your budget monthly to make adjustments.", "Try the 50/30/20 rule: 50% needs, 30% wants, 20% savings."]},

    {"tag": "investing", "patterns": ["Can you explain investing?", "How should I start investing?", "Give me tips for investing", "Where should I invest my money?"],
    "responses": ["Start by learning the basics of stocks, bonds, and mutual funds.", "Consider investing in ETFs for a low-cost and diversified portfolio.", "It's always a good idea to consult a financial advisor before making any investments."]},

    {"tag": "study_tips", "patterns": ["How can I study better?", "Give me some study tips", "I need help with studying", "How do I focus better while studying?"],
    "responses": ["Create a study schedule and stick to it.", "Break your study time into manageable chunks and take breaks in between.", "Use active recall and spaced repetition to improve memory retention."]},

    {"tag": "learning_resources", "patterns": ["Can you suggest learning resources?", "Where can I learn new skills?", "Give me resources to learn online", "Tell me about free learning resources"],
    "responses": ["Websites like Coursera, edX, and Udemy offer excellent online courses.", "You can also check out YouTube tutorials, blogs, and podcasts for free resources.", "Don’t forget about free courses offered by universities, including MIT OpenCourseWare."]},

    {"tag": "goal_setting", "patterns": ["How do I set goals?", "Give me tips on setting goals", "Help me achieve my goals", "What should my personal goals be?"],
    "responses": ["Make your goals SMART: Specific, Measurable, Achievable, Relevant, and Time-bound.", "Start with small, achievable goals, then gradually work towards bigger ones.", "Break your long-term goals into smaller, actionable steps to make progress."]},

    {"tag": "habits", "patterns": ["How do I build good habits?", "Give me tips for forming positive habits", "How can I break bad habits?", "What habits should I adopt for personal growth?"],
    "responses": ["Start small and be consistent. It's easier to build habits with gradual progress.", "Use reminders and rewards to reinforce new habits.", "Identify triggers for bad habits and replace them with better alternatives."]},

    {"tag": "healthy_lifestyle", "patterns": ["How can I live a healthy life?", "Give me tips for a healthy lifestyle", "What are some healthy habits?", "How do I stay healthy every day?"],
     "responses": ["Eat a balanced diet, exercise regularly, and get enough sleep.", "Avoid processed foods and focus on whole foods like fruits and vegetables.", "Drink plenty of water, stay active, and manage stress effectively."]},
    
    {"tag": "mental_wellness", "patterns": ["How do I improve my mental wellness?", "Give me tips for mental health", "How can I be mentally healthy?", "What are some mental wellness practices?"],
     "responses": ["Practice mindfulness and meditation.", "Engage in hobbies and activities you enjoy.", "Seek therapy or talk to someone when you're feeling down."]},
    
    {"tag": "sleep_tips", "patterns": ["How can I sleep better?", "Give me advice for good sleep", "What can I do to sleep faster?", "How do I improve my sleep quality?"],
     "responses": ["Set a regular sleep schedule and stick to it.", "Avoid caffeine and heavy meals before bed.", "Create a relaxing bedtime routine, like reading or listening to calming music."]},

    {"tag": "workout_plan", "patterns": ["What is a good workout plan?", "Can you suggest a fitness routine?", "Give me a workout plan", "What exercises should I do to get in shape?"],
     "responses": ["Try a combination of strength training and cardio.", "You can start with 3 days a week of weight training and 2 days of cardio.", "Don't forget to include flexibility exercises like yoga or stretching!"]},

    {"tag": "weight_loss", "patterns": ["How do I lose weight?", "Give me weight loss tips", "What is the best way to lose weight?", "Help me with weight loss"],
     "responses": ["Eat fewer calories than you burn, and focus on whole, nutrient-dense foods.", "Incorporate both cardio and strength training into your workout routine.", "Stay consistent, drink water, and avoid processed sugars."]},

    {"tag": "dieting", "patterns": ["What diet should I follow?", "Can you recommend a diet plan?", "Give me a diet plan", "What is the best diet for losing weight?"],
     "responses": ["Try a balanced approach with vegetables, lean proteins, and whole grains.", "Consider a low-carb or Mediterranean diet, depending on your goals.", "Focus on portion control and limit sugar and processed foods."]},

    {"tag": "fitness_goals", "patterns": ["How do I set fitness goals?", "Give me tips for setting fitness goals", "How should I track my fitness progress?", "Help me set fitness targets"],
     "responses": ["Set SMART goals: Specific, Measurable, Achievable, Relevant, and Time-bound.", "Track your progress regularly and celebrate small milestones.", "Start with attainable goals and gradually increase the challenge."]},

    {"tag": "exercise_routine", "patterns": ["Can you suggest an exercise routine?", "Give me a simple workout routine", "What should I do for a workout?", "Tell me an exercise routine"],
     "responses": ["Start with bodyweight exercises like squats, push-ups, and lunges.", "You can do a 30-minute cardio session followed by 15 minutes of strength training.", "For a full-body workout, alternate between strength and cardio exercises."]},

    {"tag": "muscle_building", "patterns": ["How do I build muscle?", "Give me tips for muscle growth", "What is the best way to gain muscle?", "Help me with muscle-building exercises"],
     "responses": ["Focus on compound movements like squats, deadlifts, and bench press.", "Eat enough protein and ensure you're in a slight calorie surplus.", "Rest and recovery are key—make sure you're allowing your muscles to recover between workouts."]},

    {"tag": "motivation_boost", "patterns": ["I need motivation", "Can you motivate me?", "Give me a motivational quote", "I'm feeling unmotivated, help me out"],
     "responses": ["“Success is not final, failure is not fatal: It is the courage to continue that counts.” – Winston Churchill", "“The only way to do great work is to love what you do.” – Steve Jobs", "“You miss 100% of the shots you don't take.” – Wayne Gretzky"]},

    {"tag": "productivity_tips", "patterns": ["How can I be more productive?", "Give me productivity tips", "How do I increase productivity?", "Help me with my productivity"],
     "responses": ["Try the Pomodoro technique: 25 minutes of focused work followed by a 5-minute break.", "Prioritize tasks and tackle the hardest one first.", "Use to-do lists and track your progress throughout the day."]},

    {"tag": "time_management", "patterns": ["How do I manage my time better?", "Give me time management tips", "What is a good time management strategy?", "How do I stay organized and manage my time?"],
     "responses": ["Use a planner or digital calendar to schedule your tasks.", "Set daily and weekly goals, and break larger tasks into smaller steps.", "Avoid multitasking and focus on one task at a time for maximum efficiency."]},

    {"tag": "focus_tips", "patterns": ["How can I improve my focus?", "Give me tips for staying focused", "How do I stay focused while working?", "What can I do to avoid distractions?"],
     "responses": ["Eliminate distractions like your phone or noisy environments.", "Take regular breaks to prevent burnout, and use techniques like Pomodoro.", "Set specific and clear goals for each task to stay focused."]},

    {"tag": "goal_setting", "patterns": ["How do I set goals?", "Give me tips on setting goals", "Help me achieve my goals", "What should my personal goals be?"],
     "responses": ["Start with small, achievable goals and break them into actionable steps.", "Make your goals SMART: Specific, Measurable, Achievable, Relevant, and Time-bound.", "Review your goals regularly and adjust them as needed."]},

    {"tag": "morning_routine", "patterns": ["What is a good morning routine?", "Give me ideas for a productive morning", "What should I do in the morning to start my day right?", "How do I start my day with energy?"],
     "responses": ["Try waking up early, drinking water, and doing some light exercise.", "Plan your day the night before and start with a healthy breakfast.", "Incorporate some mindfulness or journaling into your morning routine."]},

    {"tag": "habit_formation", "patterns": ["How do I form good habits?", "Give me tips for building habits", "What is the best way to stick to new habits?", "Help me create positive habits"],
     "responses": ["Start small and be consistent.", "Use reminders and rewards to reinforce your new habits.", "Track your progress and make adjustments if needed."]},

    {"tag": "overcoming_procrastination", "patterns": ["How do I stop procrastinating?", "Give me tips for overcoming procrastination", "What can I do to beat procrastination?", "Help me stop putting things off"],
     "responses": ["Break tasks into smaller, manageable parts and start with the easiest one.", "Set deadlines for yourself and hold yourself accountable.", "Minimize distractions and set specific times to work on tasks."]},

    {"tag": "positive_thinking", "patterns": ["How can I think positively?", "Give me tips for positive thinking", "What should I do to stay positive?", "Help me stay optimistic"],
     "responses": ["Focus on solutions, not problems. Cultivate gratitude every day.", "Surround yourself with positive influences and people.", "Challenge negative thoughts and replace them with more empowering ones."]},

    {"tag": "self_discipline", "patterns": ["How do I build self-discipline?", "Give me tips for improving self-discipline", "What should I do to stay disciplined?", "Help me improve my self-control"],
     "responses": ["Set clear goals and reward yourself when you achieve them.", "Break large tasks into small, manageable steps and stay focused.", "Create a routine and stick to it consistently."]},

    {"tag": "latest_technology", "patterns": ["What are the latest technological advancements?", "Tell me about new tech trends", "Give me news about the latest technology", "What's the latest in technology?"],
     "responses": ["Artificial intelligence and machine learning are transforming many industries.", "Quantum computing is on the rise and promises to revolutionize computing.", "5G technology is expanding rapidly, enabling faster communication and new opportunities."]},

    {"tag": "artificial_intelligence", "patterns": ["What is AI?", "Tell me about artificial intelligence", "How does AI work?", "What is the future of AI?"],
     "responses": ["AI refers to machines that can perform tasks that would normally require human intelligence.", "It uses algorithms and large data sets to make decisions and predictions.", "AI is expected to have a huge impact on healthcare, transportation, and many other fields."]},

    {"tag": "smart_home", "patterns": ["What is a smart home?", "Tell me about smart home devices", "How do I create a smart home?", "What smart home technology should I buy?"],
     "responses": ["A smart home uses internet-connected devices to manage tasks like lighting, temperature, and security.", "Popular smart devices include Alexa, Google Home, smart thermostats, and security cameras.", "You can start by installing a smart thermostat and smart light bulbs for convenience."]},

    {"tag": "cryptocurrency", "patterns": ["What is cryptocurrency?", "Tell me about Bitcoin", "How does cryptocurrency work?", "Is cryptocurrency safe?"],
     "responses": ["Cryptocurrency is a digital or virtual form of money that uses cryptography for security.", "Bitcoin is the first and most well-known cryptocurrency.", "Cryptocurrencies are decentralized and can be traded or used as investments."]},
        {"tag": "cloud_computing", "patterns": ["What is cloud computing?", "Give me information about cloud services", "How does cloud computing work?", "What are the benefits of cloud computing?"],
     "responses": ["Cloud computing allows you to access and store data and applications over the internet, rather than on your personal computer.", "Popular cloud services include AWS (Amazon Web Services), Google Cloud, and Microsoft Azure.", "Cloud computing offers scalability, cost-effectiveness, and remote access to services."]},

    {"tag": "internet_of_things", "patterns": ["What is the Internet of Things (IoT)?", "Tell me about IoT", "How does the Internet of Things work?", "What devices are part of IoT?"],
     "responses": ["IoT refers to the interconnection of everyday objects through the internet, allowing them to send and receive data.", "Smart devices like thermostats, lights, and even refrigerators are part of IoT.", "IoT can be used to improve efficiency, convenience, and data collection in various industries."]},

    {"tag": "blockchain", "patterns": ["What is blockchain?", "How does blockchain work?", "Tell me about blockchain technology", "What are the benefits of blockchain?"],
     "responses": ["Blockchain is a decentralized digital ledger used to record transactions across multiple computers.", "It ensures data integrity and security without the need for a trusted intermediary.", "Blockchain has applications in cryptocurrencies, supply chain management, and more."]},

    {"tag": "5G", "patterns": ["What is 5G?", "Tell me about 5G technology", "How does 5G work?", "What are the benefits of 5G?"],
     "responses": ["5G is the fifth generation of mobile network technology, offering faster speeds and lower latency than 4G.", "It enables more efficient communication, enhanced mobile broadband, and the growth of IoT.", "5G promises to support applications like autonomous vehicles, smart cities, and advanced healthcare."]},

    {"tag": "augmented_reality", "patterns": ["What is augmented reality?", "How does augmented reality work?", "Tell me about AR", "What are some uses of augmented reality?"],
     "responses": ["Augmented Reality (AR) overlays digital content on the real world using devices like smartphones or AR glasses.", "Popular AR applications include Pokémon Go and Snapchat filters.", "AR has potential in industries like gaming, healthcare, education, and retail."]},

    {"tag": "virtual_reality", "patterns": ["What is virtual reality?", "How does VR work?", "Tell me about virtual reality", "What are some uses of virtual reality?"],
     "responses": ["Virtual Reality (VR) creates a fully immersive digital environment, often using headsets and motion controllers.", "VR is used in gaming, education, training simulations, and therapeutic applications.", "It allows users to experience a simulated world, separate from their physical environment."]},

    {"tag": "personal_finance", "patterns": ["How can I manage my finances?", "Give me personal finance tips", "What are good personal finance practices?", "How do I improve my financial situation?"],
     "responses": ["Track your expenses and create a budget to control your spending.", "Start saving early and create an emergency fund to cover unexpected costs.", "Consider investing in a retirement account like an IRA or 401(k) for long-term growth."]},

    {"tag": "investing_basics", "patterns": ["What is investing?", "How do I start investing?", "Give me tips for investing", "What should I invest in?"],
     "responses": ["Investing involves putting your money into assets like stocks, bonds, or real estate to grow wealth over time.", "Start by researching stocks, mutual funds, and ETFs.", "Diversify your investments to reduce risk, and consider speaking with a financial advisor."]},

    {"tag": "cryptocurrency_investing", "patterns": ["Should I invest in cryptocurrency?", "Give me tips for investing in crypto", "How do I buy Bitcoin?", "What is the best cryptocurrency to invest in?"],
     "responses": ["Cryptocurrency is highly volatile and should be approached with caution.", "Start by researching major cryptocurrencies like Bitcoin, Ethereum, and Litecoin.", "Use a reputable exchange and ensure your crypto assets are stored securely in a wallet."]},

    {"tag": "stock_market", "patterns": ["How does the stock market work?", "Tell me about the stock market", "What is stock trading?", "How do I start trading stocks?"],
     "responses": ["The stock market allows you to buy and sell ownership stakes in companies (stocks).", "Stock prices are influenced by factors like company performance, economic conditions, and market sentiment.", "You can start by opening a brokerage account and researching stocks you're interested in."]},

    {"tag": "real_estate_investing", "patterns": ["How can I invest in real estate?", "Give me tips for real estate investing", "What are the benefits of real estate investing?", "How do I get started with real estate?"],
     "responses": ["Real estate investing involves buying properties to generate income through rent or appreciation.", "Start by researching the local market and understanding the costs associated with owning property.", "Consider REITs (Real Estate Investment Trusts) as a way to invest without owning property directly."]},

    {"tag": "healthy_recipes", "patterns": ["Can you recommend some healthy recipes?", "Give me healthy meal ideas", "What are some healthy foods to cook?", "How do I cook a healthy meal?"],
     "responses": ["Try grilling vegetables with olive oil and herbs for a healthy side dish.", "You can make a simple salad with mixed greens, chickpeas, and a light vinaigrette dressing.", "A quinoa and vegetable stir-fry is a healthy and filling meal option."]},

    {"tag": "easy_recipes", "patterns": ["Give me easy recipes", "What are some quick and easy meals?", "Can you suggest a simple recipe?", "I need fast meal ideas"],
     "responses": ["Make a quick pasta with tomato sauce and grilled chicken for a simple meal.", "Try scrambled eggs with spinach and avocado for a healthy breakfast.", "Make a smoothie with fruits, yogurt, and spinach for a quick meal replacement."]},

    {"tag": "vegetarian_recipes", "patterns": ["Can you suggest vegetarian recipes?", "Give me vegetarian meal ideas", "What are some good vegetarian dishes?", "How do I cook a vegetarian meal?"],
     "responses": ["You can make a veggie stir-fry with tofu, peppers, onions, and soy sauce.", "Try a hearty vegetable soup with lentils, carrots, and spinach.", "A vegetarian chili with beans, tomatoes, and spices is a filling and nutritious meal."]},

    {"tag": "dessert_recipes", "patterns": ["Can you recommend some dessert recipes?", "What are some easy desserts?", "Give me dessert ideas", "How do I make a quick dessert?"],
     "responses": ["Try making chocolate avocado mousse for a creamy, healthy dessert.", "You can bake quick chocolate chip cookies with just a few ingredients.", "Fruit salad with whipped cream or Greek yogurt is a light and delicious dessert."]},

    {"tag": "meal_prep", "patterns": ["What are some meal prep ideas?", "Give me tips for meal prepping", "How do I prepare meals for the week?", "Can you suggest meal prep recipes?"],
     "responses": ["Start by preparing grains like rice or quinoa and protein sources like chicken or beans for the week.", "Roast a variety of vegetables and portion them into containers for easy lunches.", "Make a large batch of soup or stew to last throughout the week."]},

    {"tag": "gardening", "patterns": ["How do I start gardening?", "Give me tips for gardening", "What should I plant in my garden?", "How can I grow my own vegetables?"],
     "responses": ["Start with easy-to-grow plants like herbs, tomatoes, and lettuce.", "Ensure your plants get enough sunlight, and water them regularly.", "Consider raised beds or container gardening if you have limited space."]},

    {"tag": "travel_tips", "patterns": ["Can you give me travel tips?", "What should I know before traveling?", "How do I prepare for a trip?", "Give me advice for first-time travelers"],
     "responses": ["Always check the weather before you pack and plan your clothing accordingly.", "Make sure to keep digital and physical copies of important documents like your passport.", "Learn a few basic phrases in the local language to help with communication."]},

    {"tag": "sustainable_living", "patterns": ["How can I live sustainably?", "Give me tips for sustainable living", "What are some eco-friendly habits?", "How do I reduce my carbon footprint?"],
     "responses": ["Use reusable bags, bottles, and containers to reduce waste.", "Consider adopting a plant-based diet to lower your environmental impact.", "Reduce energy consumption by switching to LED light bulbs and unplugging electronics when not in use."]},

    {"tag": "DIY_projects", "patterns": ["What are some DIY projects I can do?", "Give me ideas for home DIY projects", "Can you recommend a fun DIY project?", "What can I make at home?"],
     "responses": ["Try making your own candles with beeswax and essential oils.", "Build a simple bookshelf or create a painted picture frame as a fun weekend project.", "You can also try making homemade bath bombs or soap as a relaxing DIY project."]},

    {"tag": "fitness_hobbies", "patterns": ["What are some fitness hobbies?", "Give me ideas for fitness activities", "What are fun ways to stay fit?", "How can I make fitness enjoyable?"],
     "responses": ["Try dancing, swimming, or hiking for a fun way to stay active.", "Consider joining a local sports team or participating in a fitness class.", "Yoga and Pilates are excellent options for a relaxing yet challenging workout."]},
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
            # Directly return the first response (not random)
            return intent["responses"][0]  # You can change this if you need a different logic

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