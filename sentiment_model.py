import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import random
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

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
    
    return 'latin-1'

def load_data(data_path):
    """Load and preprocess the dataset"""
    try:
        with open("encoding_config.txt", "r") as f:
            encoding = f.read().strip()
    except:
        encoding = detect_encoding(data_path)
    
    # Try different parsing engines
    for engine in ['python', 'c']:
        try:
            df = pd.read_csv(data_path, encoding=encoding, engine=engine)
            break
        except:
            df = pd.read_csv(data_path, encoding=encoding, on_bad_lines='skip')
    
    # Auto-detect text and sentiment columns
    text_col = None
    sentiment_col = None
    
    # Common column name patterns
    text_candidates = ['text', 'message', 'content', 'tweet', 'review']
    sentiment_candidates = ['sentiment', 'label', 'class', 'emotion', 'polarity']
    
    for col in df.columns:
        if col.lower() in text_candidates:
            text_col = col
        elif col.lower() in sentiment_candidates:
            sentiment_col = col
    
    # Fallback to first string column for text
    if not text_col:
        for col in df.columns:
            if df[col].dtype == 'object':
                text_col = col
                break
    
    # Fallback to limited-value column for sentiment
    if not sentiment_col:
        for col in df.columns:
            if col != text_col and len(df[col].unique()) <= 10:
                sentiment_col = col
                break
    
    if not text_col or not sentiment_col:
        raise ValueError("Could not identify text and sentiment columns")
    
    return df[text_col], df[sentiment_col]

def train_sentiment_model(data_path='text.csv'):
    """Train a sentiment analysis model targeting ~80% accuracy"""
    try:
        # Load and preprocess data
        X, y = load_data(data_path)
        
        # Print data stats
        print(f"\nDataset size: {len(X)}")
        print("Class distribution:")
        print(y.value_counts())
        
        # Stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Feature engineering pipeline
        vectorizer = CountVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Use unigrams and bigrams
            max_features=4000,    # Increased feature limit
            min_df=3             # Minimum document frequency
        )

        # Classifiers with optimized hyperparameters
        classifiers = {
            'logistic': LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight='balanced',
                solver='liblinear',
                penalty='l2'
            ),
            'svm': LinearSVC(
                C=1.0,
                class_weight='balanced',
                dual=False,
                max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=25,
                min_samples_leaf=2,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5
            )
        }

        # Train and evaluate models
        best_model = None
        best_score = 0
        best_name = ""

        for name, classifier in classifiers.items():
            print(f"\nTraining {name} classifier...")
            
            pipeline = make_pipeline(
                vectorizer,
                TfidfTransformer(),
                SMOTE(random_state=42),  # Handle class imbalance
                classifier
            )
            
            pipeline.fit(X_train, y_train)
            score = pipeline.score(X_test, y_test)
            print(f"{name} accuracy: {score:.2%}")
            
            if score > best_score:
                best_score = score
                best_model = pipeline
                best_name = name

        # Generate detailed metrics
        y_pred = best_model.predict(X_test)
        accuracy = best_score
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Print final metrics
        print("\n" + "="*50)
        print(f"Best model: {best_name} (Accuracy: {accuracy:.2%})")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(conf_matrix)

        # Save model and metrics
        with open('sentiment_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)

        metrics = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'classes': list(np.unique(y)),
            'best_classifier': best_name
        }

        with open('model_metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)

        return best_model, accuracy, metrics

    except Exception as e:
        print(f"Error in training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def load_model(model_path='sentiment_model.pkl'):
    """Load the trained sentiment model"""
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    print("Model not found. Train a new model first.")
    return None

def load_model_metrics():
    """Load the saved model metrics"""
    if os.path.exists('model_metrics.pkl'):
        with open('model_metrics.pkl', 'rb') as f:
            return pickle.load(f)
    return None

def predict_sentiment(texts, model=None):
    """Predict sentiment for a list of texts"""
    if model is None:
        model = load_model()
    
    if model is None:
        return ['neutral'] * len(texts)
    
    try:
        predictions = model.predict(texts)
        
        # Get probabilities if available
        try:
            proba = model.predict_proba(texts)
            class_names = model.classes_
            
            results = []
            for i, pred in enumerate(predictions):
                scores = proba[i]
                class_probs = {class_name: scores[j] for j, class_name in enumerate(class_names)}
                results.append({
                    'sentiment': pred,
                    'probabilities': class_probs,
                    'confidence': max(scores)
                })
            return results
            
        except:
            # Fallback for classifiers without predict_proba
            return [{'sentiment': pred, 'probabilities': None, 'confidence': None} 
                   for pred in predictions]
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return ['neutral'] * len(texts)

def get_model_performance_summary():
    """Get a text summary of model performance"""
    metrics = load_model_metrics()
    if not metrics:
        return "No model metrics available"
    
    summary = f"Best Model: {metrics.get('best_classifier', 'N/A')}\n"
    summary += f"Accuracy: {metrics.get('accuracy', 0):.2%}\n\n"
    
    report = metrics.get('report', {})
    if report:
        summary += "Class-wise Performance:\n"
        for class_name, scores in report.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            summary += (f"- {class_name}: "
                       f"Precision={scores['precision']:.2f}, "
                       f"Recall={scores['recall']:.2f}, "
                       f"F1={scores['f1-score']:.2f}\n")
    
    return summary