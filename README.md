# Phase-wise NLP Analysis with Model Comparison

This is a **Streamlit web app** for **Fake vs Real News detection (and other classification tasks)** using **phase-wise NLP feature extraction**.  
It allows you to experiment with multiple **linguistic phases** and compare the performance of **different ML models**.

---

## Features

- Upload any CSV dataset  
- Select text & target columns dynamically  
- Choose from **5 NLP Phases**:
  - Lexical & Morphological (tokenization, lemmatization, stopword removal)  
  - Syntactic (POS tags)  
  - Semantic (sentiment polarity & subjectivity)  
  - Discourse (sentence relations, length features)  
  - Pragmatic (modality words, interrogatives, exclamations)  
- Trains and evaluates 4 ML models automatically:
  - Naive Bayes  
  - Decision Tree  
  - Logistic Regression  
  - SVM  
- Visualizations:
  - Accuracy bar chart (highlighting best model)  
  - Performance distribution pie chart  
  - Interactive metrics dashboard  
- Ranked table of model performance  

---

## 📂 Project Structure

