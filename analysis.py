import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import re
from sentiment_model import load_model, predict_sentiment

def get_message_counts(df):
    return df['sender'].value_counts()

def get_daily_activity(df):
    df['date'] = df['datetime'].dt.date
    return df.groupby('date').count()['message']

def get_common_words(df, num=10):
    all_words = ' '.join(df['message']).lower()
    words = re.findall(r'\b\w+\b', all_words)
    return Counter(words).most_common(num)

def generate_wordcloud(df):
    text = ' '.join(df['message']).lower()
    return WordCloud(width=800, height=400, background_color='white').generate(text)

def analyze_sentiment(df):
    """Analyze sentiment of messages using custom trained model"""
    # Load the sentiment model
    try:
        model = load_model()
        
        # Get sentiment predictions
        predictions = predict_sentiment(df['message'].tolist(), model)
        
        # Extract sentiment labels and scores
        sentiments = []
        for pred in predictions:
            # Check if pred is a dictionary (as expected) or a string
            if isinstance(pred, dict):
                sentiment = pred['sentiment']
                probs = pred['probabilities']
                
                # Get probability scores
                positive_score = probs.get('positive', 0)
                negative_score = probs.get('negative', 0)
                neutral_score = probs.get('neutral', 0)
                
                # Calculate compound score
                compound_score = positive_score - negative_score
            else:
                # If prediction is just a string (e.g., 'positive', 'negative', 'neutral')
                sentiment = pred
                # Set default values for scores
                positive_score = 1.0 if sentiment == 'positive' else 0.0
                negative_score = 1.0 if sentiment == 'negative' else 0.0
                neutral_score = 1.0 if sentiment == 'neutral' else 0.0
                compound_score = 1.0 if sentiment == 'positive' else (-1.0 if sentiment == 'negative' else 0.0)
            
            sentiments.append({
                'sentiment': sentiment,
                'positive': positive_score,
                'negative': negative_score,
                'neutral': neutral_score,
                'compound': compound_score
            })
        
        # Convert to DataFrame
        sentiment_df = pd.DataFrame(sentiments)
        
        # Combine with original dataframe
        result_df = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
        
        return result_df
    except Exception as e:
        # If there's an error, return the original dataframe with default sentiment values
        print(f"Error in sentiment analysis: {e}")
        df['sentiment'] = 'neutral'
        df['positive'] = 0.0
        df['negative'] = 0.0
        df['neutral'] = 1.0
        df['compound'] = 0.0
        return df

def get_sentiment_by_sender(df):
    """Get average sentiment scores grouped by sender"""
    sentiment_df = analyze_sentiment(df)
    return sentiment_df.groupby('sender')[['compound', 'positive', 'negative', 'neutral']].mean()

def get_sentiment_over_time(df):
    """Get sentiment trends over time"""
    sentiment_df = analyze_sentiment(df)
    sentiment_df['date'] = sentiment_df['datetime'].dt.date
    return sentiment_df.groupby('date')['compound'].mean()