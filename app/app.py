import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download NLTK data
@st.cache_resource
def download_nltk_data():
  nltk.download('stopwords', quiet=True)

download_nltk_data()

# Load the pre-trained model and vectorizer
@st.cache_resource
def load_model():
  model = pickle.load(open('trained_model.save', 'rb'))
  vectorizer = pickle.load(open('vectorizer.save', 'rb'))
  return model, vectorizer

loaded_model, loaded_vectorizer = load_model()

# Initialize PorterStemmer
port_stem = PorterStemmer()

def stemming(content):
  stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content

def predict_sentiment(text):
  processed_input = stemming(text)
  vectorized_input = loaded_vectorizer.transform([processed_input])
  prediction = loaded_model.predict(vectorized_input)
  return 'Positive' if prediction[0] == 1 else 'Negative'

def create_wordcloud(text, title):
  wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
  fig, ax = plt.subplots(figsize=(10, 5))
  ax.imshow(wordcloud, interpolation='bilinear')
  ax.axis('off')
  ax.set_title(title)
  return fig

# Sample tweets
sample_tweets = [
  "The sunset tonight is absolutely breathtaking. Nature never fails to amaze me. 🌅",
  "My best friend surprised me with tickets to my favorite band's concert. Can't wait! #BestFriendEver",
  "Stuck in traffic for 2 hours. This commute is ruining my life. 😡 #TrafficHell",
  "Just found out I didn't get the job I really wanted. Feeling so disappointed right now.",
  "My flight got cancelled and now I'm stranded at the airport. Worst travel experience ever. #Frustrated"
]

def single_tweet_analysis():
  st.header("Single Tweet Analysis")

  # Dropdown for sample tweets
  st.subheader("Sample Tweets")
  sample_tweet = st.selectbox("Select a sample tweet:", [""] + sample_tweets)
  
  if sample_tweet:
      if st.button('Analyze Sample Tweet'):
          sentiment = predict_sentiment(sample_tweet)
          st.success(f'The Sample Tweet is {sentiment}')
  # Custom search box
  st.subheader("Custom Tweet")
  custom_tweet = st.text_area("Enter your own tweet:")
  
  if st.button('Analyze Custom Tweet'):
      if custom_tweet:
          sentiment = predict_sentiment(custom_tweet)
          st.success(f'The Sample Tweet is {sentiment}')
      else:
          st.warning('Please enter a tweet to analyze.')

def multiple_tweets_analysis():
  st.header("Multiple Tweets Analysis")

  tweets = st.text_area("Enter multiple tweets (one per line):")
  
  if st.button('Analyze Tweets'):
      if tweets:
          tweet_list = tweets.split('\n')
          results = []
          positive_words = []
          negative_words = []

          for tweet in tweet_list:
              if tweet.strip():  # Check if the tweet is not empty
                  sentiment = predict_sentiment(tweet)
                  results.append(sentiment)
                  words = tweet.lower().split()
                  if sentiment == 'Positive':
                      positive_words.extend(words)
                  else:
                      negative_words.extend(words)

          # Display results
          st.subheader("Analysis Results:")
          for tweet, sentiment in zip(tweet_list, results):
              if tweet.strip():
                  st.write(f"{'✅' if sentiment == 'Positive' else '❌'} {sentiment}: {tweet}")

          # Create word clouds
          col1, col2 = st.columns(2)
          with col1:
              if positive_words:
                  st.subheader('Positive Word Cloud')
                  st.pyplot(create_wordcloud(' '.join(positive_words), 'Positive Words'))
          with col2:
              if negative_words:
                  st.subheader('Negative Word Cloud')
                  st.pyplot(create_wordcloud(' '.join(negative_words), 'Negative Words'))

      else:
          st.warning('Please enter tweets to analyze.')

def main():
  st.title('Tweet Sentiment Analyzer')

  # Sidebar
  analysis_type = st.sidebar.radio("Choose analysis type:", ("Single Tweet", "Multiple Tweets"))

  if analysis_type == "Single Tweet":
      single_tweet_analysis()
  else:
      multiple_tweets_analysis()

if __name__ == '__main__':
  main()