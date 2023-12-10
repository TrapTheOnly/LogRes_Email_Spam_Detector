import pandas as pd
import numpy as np
import re
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Preprocessing function for text data
def clean_text(text):
    """
    Cleans the text data by lowercasing, removing special characters, and stopwords, then applies stemming.

    Parameters:
    text (str): The text to be cleaned.

    Returns:
    str: The cleaned and stemmed text.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    cleaned_text = " ".join([word for word in words if word not in stop_words])
    stemmer = PorterStemmer()
    stemmed_text = " ".join([stemmer.stem(word) for word in cleaned_text.split()])
    return stemmed_text

# Loading and preprocessing the dataset
def load_and_preprocess_data(file_path):
    """
    Loads the dataset and applies preprocessing steps.

    Parameters:
    file_path (str): The file path of the dataset.

    Returns:
    DataFrame: The preprocessed dataset.
    """
    data = pd.read_csv(file_path)
    data['text'] = data['text'].apply(clean_text)
    data['subject'] = data['text'].apply(lambda x: x.split('\r\n')[0][9:])
    data['is_response'] = data['subject'].apply(lambda sub: 1 if any(kw in sub.lower() for kw in ['re:', 'fw:', 'fwd:']) else 0)
    return data

# Function to train the spam detection model
def train_spam_detection_model(data):
    """
    Trains the spam detection model using Logistic Regression.

    Parameters:
    data (DataFrame): The preprocessed dataset.

    Returns:
    LogisticRegression: The trained model.
    TfidfVectorizer: The vectorizer used for feature transformation.
    """
    # Splitting the dataset
    X = data[['text', 'is_response']]
    y = data['label_num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Vectorizing the text
    vectorizer = TfidfVectorizer()
    X_train_transformed = vectorizer.fit_transform(X_train['text'])
    X_test_transformed = vectorizer.transform(X_test['text'])

    # Include 'is_response' feature
    X_train_transformed = scipy.sparse.hstack((X_train_transformed, np.array(X_train['is_response'])[:, None]))
    X_test_transformed = scipy.sparse.hstack((X_test_transformed, np.array(X_test['is_response'])[:, None]))

    # Training the Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_transformed, y_train)
    
    # Evaluating the model
    y_pred = model.predict(X_test_transformed)
    print(classification_report(y_test, y_pred))

    # Cross-validation
    scores = cross_val_score(model, X_train_transformed, y_train, cv=5)
    print(f'Average Cross-Validated Accuracy: {scores.mean():.4f}\n')
    
    return model, vectorizer

# Function to get email input from the user
def get_email_input():
    """
    Prompts the user to input an email's subject and body.

    Returns:
    str: The subject of the email.
    str: The body of the email.
    """
    subject = input("Enter email subject: ")
    print("Enter email body (type 'END' on a new line to finish):")
    body_lines = []
    while True:
        line = input()
        if line == 'END':
            break
        body_lines.append(line)
    body = "\n".join(body_lines)
    return subject, body

# Function to preprocess and predict whether an email is spam
def predict_spam(subject, body, model, vectorizer):
    """
    Predicts whether an email is spam or not.

    Parameters:
    subject (str): The subject of the email.
    body (str): The body of the email.
    model (LogisticRegression): The trained spam detection model.
    vectorizer (TfidfVectorizer): The vectorizer for feature transformation.

    Returns:
    int: Prediction result where 0 is 'Not Spam' and 1 is 'Spam'.
    """
    # Preprocess email
    features = preprocess_email(subject, body, vectorizer)

    # Make a prediction
    prediction = model.predict(features)[0]
    return prediction

# Function to preprocess email for prediction
def preprocess_email(subject, body, vectorizer):
    """
    Preprocesses the email for prediction.

    Parameters:
    subject (str): The subject of the email.
    body (str): The body of the email.
    vectorizer (TfidfVectorizer): The vectorizer for feature transformation.

    Returns:
    sparse matrix: The transformed features of the email.
    """
    subject_clean = clean_text(subject)
    body_clean = clean_text(body)
    is_response = 1 if any(kw in subject.lower() for kw in ['re:', 'fw:', 'fwd:']) else 0

    subject_features = vectorizer.transform([subject_clean])
    body_features = vectorizer.transform([body_clean])

    # Combine features
    features = scipy.sparse.hstack((body_features, np.array([[is_response]])))
    return features

# Main execution
if __name__ == "__main__":
    # Load and preprocess the data
    data = load_and_preprocess_data('spam_ham_dataset.csv')

    # Train the model
    model, vectorizer = train_spam_detection_model(data)

    while True:
        # Get user input
        subject, body = get_email_input()

        # Predict spam or not
        prediction = predict_spam(subject, body, model, vectorizer)

        if prediction == 0:
            print("The email is classified as Not Spam.")
        else:
            print("The email is classified as Spam.")

        # Ask the user if they want to continue
        continue_prediction = input("Do you want to check another email? (yes/no): ").lower()
        if continue_prediction != 'yes':
            break

    print("Exiting program...")
