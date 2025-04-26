import traceback
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import traceback
from parser import parse_chat
from analysis import (
    get_message_counts, get_daily_activity, get_common_words, 
    generate_wordcloud, analyze_sentiment, get_sentiment_by_sender, 
    get_sentiment_over_time
)
from sentiment_model import train_sentiment_model, load_model_metrics

def detect_encoding(file_path):
    """Try to detect the encoding of a file"""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'utf-16', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read()
                return encoding
        except UnicodeDecodeError:
            continue
    
    # If no encoding worked, return a default
    return 'latin-1'  # latin-1 should read any file without raising errors

# Function to display model metrics visualization
def display_model_metrics():
    metrics = load_model_metrics()
    if metrics:
        st.subheader("Model Performance Metrics")
        
        # Display overall accuracy
        accuracy = metrics.get('accuracy', 0)
        st.metric("Overall Accuracy", f"{accuracy:.2%}")
        
        # Display detailed metrics
        report = metrics.get('report', {})
        
        # Create a DataFrame for the classification report
        if report:
            # Extract relevant metrics
            classes = []
            precisions = []
            recalls = []
            f1_scores = []
            supports = []
            
            for class_name, metrics_dict in report.items():
                if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                classes.append(class_name)
                precisions.append(metrics_dict['precision'])
                recalls.append(metrics_dict['recall'])
                f1_scores.append(metrics_dict['f1-score'])
                supports.append(metrics_dict['support'])
            
            metrics_df = pd.DataFrame({
                'Class': classes,
                'Precision': precisions,
                'Recall': recalls,
                'F1 Score': f1_scores,
                'Support': supports
            })
            
            # Display metrics table
            st.write("Performance by Class:")
            st.dataframe(metrics_df)
            
            # Create a bar chart
            st.write("F1 Scores by Class:")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(classes, f1_scores, color='skyblue')
            ax.set_ylim(0, 1)
            ax.set_ylabel('F1 Score')
            ax.set_title('Model Performance (F1 Score) by Class')
            st.pyplot(fig)
            
            # Display confusion matrix
            st.write("Confusion Matrix:")
            conf_matrix = metrics.get('confusion_matrix', [])
            class_names = metrics.get('classes', classes)
            
            if conf_matrix:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=class_names, yticklabels=class_names, ax=ax)
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
    else:
        st.info("No model metrics available. Train a model first to see performance metrics.")

st.set_page_config(page_title="MidTrack - WhatsApp Chat Analyzer", layout="wide")
st.title("📱 MidTrack - WhatsApp Chat Analyzer")

# Sidebar for model training
with st.sidebar:
    st.header("Sentiment Analysis Model")
    st.write("Upload a dataset to train the sentiment analysis model")
    dataset_file = st.file_uploader("Upload sentiment dataset (CSV)", type="csv")
    
    if dataset_file is not None:
        # Save the uploaded file temporarily
        with open("text.csv", "wb") as f:
            f.write(dataset_file.getbuffer())
        
        # Display dataset preview
        try:
            # Try to detect encoding
            encoding = detect_encoding("text.csv")
            st.write(f"Detected file encoding: {encoding}")
            
            # Try to read with detected encoding
            preview_df = pd.read_csv("text.csv", nrows=5, encoding=encoding)
            st.write("Dataset Preview:")
            st.dataframe(preview_df)
            
            # Show column info to help user verify correct columns
            st.write("Dataset Columns:")
            st.write(", ".join(preview_df.columns.tolist()))
            
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            st.write("Trying alternative approach...")
            try:
                # Fallback to latin-1 encoding which usually works for most files
                preview_df = pd.read_csv("text.csv", nrows=5, encoding='latin-1', on_bad_lines='skip')
                st.write("Dataset Preview (using fallback encoding):")
                st.dataframe(preview_df)
            except Exception as e2:
                st.error(f"Still cannot read the file: {e2}")
        
        # Add encoding options
        encoding_options = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        selected_encoding = st.selectbox("Select encoding if automatic detection failed:", encoding_options)
        
        if st.button("Train Sentiment Model"):
            with st.spinner("Training model..."):
                try:
                    # Add the encoding to a config file for the training function to use
                    with open("encoding_config.txt", "w") as f:
                        f.write(selected_encoding)
                    
                    result = train_sentiment_model('text.csv')
                    if result:
                        model, accuracy, metrics = result
                        st.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")
                        
                        # Display metrics summary
                        if metrics and 'report' in metrics:
                            report = metrics['report']
                            st.write("Performance by class:")
                            for class_name, metrics_dict in report.items():
                                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                                    st.write(f"- {class_name}: F1-score = {metrics_dict['f1-score']:.2f}")
                    else:
                        st.error("Failed to train model. Check the console for details.")
                except Exception as e:
                    st.error(f"Error training model: {e}")
                    st.code(traceback.format_exc())
    
    # Add Model Performance section in sidebar
    st.header("Model Performance")
    if st.button("View Detailed Model Metrics"):
        display_model_metrics()

# Main app
uploaded_file = st.file_uploader("Upload WhatsApp chat export (.txt file without media)", type="txt")

if uploaded_file is not None:
    chat_text = uploaded_file.read().decode("utf-8")
    df = parse_chat(chat_text)

    if not df.empty:
        st.success("✅ Chat successfully parsed!")

        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Basic Stats", "Word Analysis", "Sentiment Analysis", "Model Accuracy", "Search"])
        
        with tab1:
            # 📊 Stats
            st.subheader("📊 Basic Stats")
            st.bar_chart(get_message_counts(df))
            st.line_chart(get_daily_activity(df))
        
        with tab2:
            # 🧠 Common Words
            st.subheader("🧠 Most Common Words")
            common_words = get_common_words(df)
            st.dataframe(pd.DataFrame(common_words, columns=["Word", "Count"]))
            st.image(generate_wordcloud(df).to_array())
        
        with tab3:
            # 😀 Sentiment Analysis
            st.subheader("😀 Sentiment Analysis")
            
            # Get sentiment analysis results
            with st.spinner("Analyzing sentiments..."):
                sentiment_df = analyze_sentiment(df)
            
            # Overall sentiment distribution
            sentiment_counts = sentiment_df['sentiment'].value_counts()
            st.write("Overall Sentiment Distribution:")
            st.bar_chart(sentiment_counts)
            
            # Sentiment by sender
            st.write("Average Sentiment by Sender:")
            sender_sentiment = get_sentiment_by_sender(df)
            st.dataframe(sender_sentiment)
            st.bar_chart(sender_sentiment['compound'])
            
            # Sentiment over time
            st.write("Sentiment Trends Over Time:")
            time_sentiment = get_sentiment_over_time(df)
            st.line_chart(time_sentiment)
            
            # Display messages with their sentiment
            st.write("Messages with Sentiment Analysis:")
            display_cols = ['datetime', 'sender', 'message', 'sentiment', 'compound']
            st.dataframe(sentiment_df[display_cols])
            
        with tab4:
            # 📈 Model Accuracy 
            st.subheader("📈 Model Accuracy Metrics")
            display_model_metrics()
            
            # Add confidence analysis
            if 'positive' in sentiment_df.columns and 'negative' in sentiment_df.columns:
                st.subheader("Sentiment Confidence Analysis")
                
                # Create confidence score (higher value = more certain)
                sentiment_df['confidence'] = sentiment_df.apply(
                    lambda row: max(row['positive'], row['negative'], row['neutral']), 
                    axis=1
                )
                
                # Plot confidence distribution
                fig, ax = plt.subplots()
                ax.hist(sentiment_df['confidence'], bins=10, alpha=0.7)
                ax.set_xlabel('Confidence Score')  
                ax.set_ylabel('Number of Messages')
                ax.set_title('Distribution of Sentiment Confidence Scores')
                st.pyplot(fig)
                
                # Display high confidence vs low confidence examples
                high_conf = sentiment_df.nlargest(5, 'confidence')[['message', 'sentiment', 'confidence']]
                low_conf = sentiment_df.nsmallest(5, 'confidence')[['message', 'sentiment', 'confidence']]
                
                st.write("Examples of High Confidence Predictions:")
                st.dataframe(high_conf)
                
                st.write("Examples of Low Confidence Predictions:")
                st.dataframe(low_conf)
        
        with tab5:
            # 🔍 Search
            st.subheader("🔍 Search Messages")
            user_filter = st.selectbox("Filter by sender", options=["All"] + list(df['sender'].unique()))
            keyword = st.text_input("Enter keyword to search (optional):")
            sentiment_filter = st.selectbox("Filter by sentiment", options=["All", "positive", "negative", "neutral"])

            filtered = df.copy()
            if user_filter != "All":
                filtered = filtered[filtered['sender'] == user_filter]
            if keyword:
                filtered = filtered[filtered['message'].str.contains(keyword, case=False)]
                
            # Apply sentiment filter if needed
            if sentiment_filter != "All":
                # First run sentiment analysis if it hasn't been done yet
                if 'sentiment' not in filtered.columns:
                    sentiment_results = analyze_sentiment(filtered)
                    filtered = sentiment_results
                
                filtered = filtered[filtered['sentiment'] == sentiment_filter]

            st.write(f"Displaying {len(filtered)} messages:")
            display_cols = ['datetime', 'sender', 'message']
            if 'sentiment' in filtered.columns:
                display_cols.append('sentiment')
            st.dataframe(filtered[display_cols])
    else:
        st.error("Could not parse the chat. Please ensure it's in the correct format.")