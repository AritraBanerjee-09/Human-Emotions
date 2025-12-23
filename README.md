# 🎭 Six Human Emotions Detection App

A Machine Learning–based web application that predicts **six human emotions** from text using **Natural Language Processing (NLP)** techniques and **Logistic Regression**, deployed using **Streamlit**.

---

## 📌 Table of Contents
1. Project Overview  
2. Problem Statement  
3. Emotions Covered  
4. Tech Stack  
5. Dataset  
6. Model & Approach  
7. Project Architecture  
8. Installation & Setup  
9. Running the Application  
10. Model Performance  
11. Code & Environment Versions  
12. Security & Access  
13. Future Enhancements  
14. Author  

---

## 📖 Project Overview

Understanding human emotions from text is a key challenge in sentiment analysis and NLP.  
This project classifies user input text into one of six emotional categories, providing both the **predicted emotion** and **confidence score** in real time through a web interface.

---

## ❓ Problem Statement

To build an efficient NLP-based system that can:
- Process raw text input
- Extract meaningful features
- Accurately classify emotions
- Provide results through an interactive UI

---

## 🎯 Emotions Covered

- Joy  
- Fear  
- Anger  
- Love  
- Sadness  
- Surprise  

---

## 🛠 Tech Stack

### Programming Language
- Python

### Libraries & Frameworks
- Streamlit – Web UI
- Scikit-learn – Machine Learning models
- NLTK – Text preprocessing
- NumPy – Numerical operations
- Regex – Text cleaning
- Pickle – Model serialization

---

## 📊 Dataset

- Source: Emotion-labeled text dataset
- Size: ~16,000 samples
- Format: Text + Emotion label
- Preprocessing:
  - Duplicate removal
  - Stopword removal
  - Stemming
  - Lowercasing

---

## 🧠 Model & Approach

### Text Preprocessing
- Regex-based cleaning
- Tokenization
- Stopword removal
- Porter stemming

### Feature Engineering
- TF-IDF Vectorization

### Models Evaluated
- Multinomial Naive Bayes  
- Logistic Regression ✅ (Selected)  
- Random Forest  
- Support Vector Machine  

### Final Model
- **Logistic Regression (Multiclass Classification)**

---

## 🏗 Project Architecture

User Input
↓
Text Cleaning & Preprocessing
↓
TF-IDF Vectorization
↓
Logistic Regression Model
↓
Emotion Prediction + Confidence