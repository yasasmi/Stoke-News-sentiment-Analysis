📈 Stock Price Prediction Using Sentiment Analysis & LSTM
Predicting stock market trends is a complex challenge influenced by numerous factors, including investor sentiment. This project combines Natural Language Processing (NLP) and Long Short-Term Memory (LSTM) networks to forecast stock price movements based on the sentiment extracted from financial news headlines.

📌 Project Overview
This deep learning pipeline:

Extracts sentiment from real-world financial headlines.

Correlates sentiment trends with historical stock price movements.

Predicts future stock direction using an RNN-LSTM model.

🚀 Key Highlights
✅ Sentiment-Driven LSTM Model

Trained an RNN-LSTM model to detect temporal patterns in sentiment-rich news data.

✅ Advanced NLP Techniques

Preprocessing: Tokenization, stopword removal, stemming, lemmatization

Feature extraction: TF-IDF, Word2Vec, GloVe

Sentiment scoring: VADER, TextBlob, BERT-based models

✅ Real-Time Financial Data Integration

Collected stock prices using Yahoo Finance API / Alpha Vantage

Scraped and cleaned news headlines from trusted financial sources (e.g., Reuters, Bloomberg)

✅ Model Optimization

Applied dropout regularization, sequence padding

Tuned hyperparameters (batch size, learning rate, sequence length)

✅ Performance Evaluation

Regression Metrics: RMSE, MAE

Classification Metrics: Precision, Recall, F1-score

✅ Visualization Tools

Used Matplotlib, Seaborn, and Plotly to visualize sentiment shifts and stock movements

✅ Interactive Deployment

Built a user-friendly web app using Flask and Streamlit for live sentiment-based trend predictions

🔧 Tech Stack
Languages/Frameworks: Python, TensorFlow/Keras, Flask, Streamlit

NLP Tools: NLTK, SpaCy, Gensim, Transformers (Hugging Face)

Data Handling: Pandas, NumPy

Visualization: Matplotlib, Seaborn, Plotly

APIs: Yahoo Finance, Alpha Vantage

