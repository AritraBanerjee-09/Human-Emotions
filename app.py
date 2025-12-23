#========================import packages=========================================================
import streamlit as st
import numpy as np
import re
from nltk.stem import PorterStemmer
import pickle
import nltk

# Download stopwords only once
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

#========================loading the saved files=================================================
lg = pickle.load(open('logistic_regresion.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
lb = pickle.load(open('label_encoder.pkl', 'rb'))

# =========================text cleaning function===============================================
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    # Predict class
    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]

    # ✅ Correct probability
    probability = np.max(lg.predict_proba(input_vectorized))

    return predicted_emotion, probability

#==================================creating app=================================================
st.title("Six Human Emotions Detection App")
st.write("=================================================")
st.write("['Joy','Fear','Anger','Love','Sadness','Surprise']")
st.write("=================================================")

user_input = st.text_input("Enter your text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        predicted_emotion, probability = predict_emotion(user_input)
        st.write("Predicted Emotion:", predicted_emotion)
        st.write("Probability:", round(probability * 100, 2), "%")
