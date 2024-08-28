# Tweet Sentiment Analyzer

A machine learning-powered web application that analyzes the sentiment of tweets using natural language processing techniques.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Technologies Used](#technologies-used)

## Project Overview

This project demonstrates the end-to-end process of building a sentiment analysis model for tweets and deploying it as a web application. It includes data preprocessing, model training, and a user-friendly interface for real-time sentiment prediction.


## Features

- Single tweet sentiment analysis
- Multiple tweet sentiment analysis
- Interactive web interface using Streamlit
- Word cloud visualization for positive and negative sentiments
- Pre-trained model for quick predictions
- Sample tweets for easy testing

## Installation

1. Clone the repository:
git clone https://github.com/simha-p/Twitter_Sentiment_Analysis.git
cd Twitter_Sentiment_Analysis

2. Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate


3. Install the required packages:
pip install -r requirements.txt


4. Download the dataset:
   - Sign up on Kaggle and download your API key (kaggle.json)
   - Place the kaggle.json file in the ~/.kaggle/ directory
   - Run the following commands:
     ```
    mkdir -p ~/.kaggle
    cp kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    kaggle datasets download -d kazanova/sentiment140
 

## Usage

1. To run the Streamlit app:
cd app
streamlit run app.py


2. Open your web browser and go to `http://localhost:8501`

3. Choose between single tweet analysis or multiple tweets analysis

4. For single tweet analysis:
   - Select a sample tweet from the dropdown or enter your own tweet
   - Click "Analyze" to see the sentiment prediction

5. For multiple tweets analysis:
   - Enter multiple tweets (one per line) in the text area
   - Click "Analyze Tweets" to see sentiment predictions and word clouds

## Model Training

The sentiment analysis model was trained using the following process:

## Demo of Application


1. Data loading and preprocessing from the Sentiment140 dataset
2. Text cleaning and tokenization using NLTK
3. Feature extraction using TF-IDF vectorization
4. Model training using Logistic Regression
5. Model evaluation and performance metrics calculation

For detailed implementation, please refer to the Jupyter Notebook in the  directory.

## Technologies Used

- Python 
- Streamlit for web application
- scikit-learn for machine learning
- NLTK for natural language processing
- pandas and NumPy for data manipulation
- Matplotlib and WordCloud for visualization

## Demo of Application
[sentimental_analysis (2).webm](https://github.com/user-attachments/assets/32ef4c1f-dda1-4a89-b285-bea4e2f11287)

