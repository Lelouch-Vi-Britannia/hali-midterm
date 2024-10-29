# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import os
import joblib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.sparse import hstack, csr_matrix
import seaborn as sns

# Download necessary NLTK data
nltk.download('stopwords')
from nltk.corpus import stopwords

# Ensure reproducibility
np.random.seed(0)

# Load Data
trainingSet = pd.read_csv("./data/train.csv")
testingSet = pd.read_csv("./data/test.csv")

print("train.csv shape is ", trainingSet.shape)
print("test.csv shape is ", testingSet.shape)

# Adding Features
def add_features_to(df):
    # Feature extraction
    df['Helpfulness'] = df['HelpfulnessNumerator'] / (df['HelpfulnessDenominator'] + 1)
    df['Helpfulness'] = df['Helpfulness'].fillna(0)

    # Text length features
    df['Summary_length'] = df['Summary'].fillna('').apply(len)
    df['Text_length'] = df['Text'].fillna('').apply(len)

    return df

# Apply feature extraction
trainingSet = add_features_to(trainingSet)

# Prepare Feature Set and Labels
features = [
    'HelpfulnessNumerator',
    'HelpfulnessDenominator',
    'Helpfulness',
    'Summary_length',
    'Text_length'
]

X = trainingSet[trainingSet['Score'].notnull()]
Y = X['Score']
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.25,
    random_state=0,
    stratify=Y
)

# Text Preprocessing
custom_stop_words = ['movie', 'film']
stop_words = stopwords.words('english') + custom_stop_words

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-letter characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stop words
    return text

# Apply text preprocessing
for df in [X_train, X_test]:
    df['Summary'] = df['Summary'].apply(preprocess_text)
    df['Text'] = df['Text'].apply(preprocess_text)

# Feature Engineering: Text Features
tfidf_summary = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words=stop_words)
tfidf_text = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words=stop_words)

# Fit and transform the text data
tfidf_summary.fit(X_train['Summary'])
tfidf_text.fit(X_train['Text'])

summary_train_tfidf = tfidf_summary.transform(X_train['Summary'])
text_train_tfidf = tfidf_text.transform(X_train['Text'])

summary_test_tfidf = tfidf_summary.transform(X_test['Summary'])
text_test_tfidf = tfidf_text.transform(X_test['Text'])

# Scale numerical features
X_train_select = X_train[features]
X_test_select = X_test[features]

scaler = StandardScaler()
X_train_select_scaled = scaler.fit_transform(X_train_select)
X_test_select_scaled = scaler.transform(X_test_select)

# Combine numerical and text features
X_train_final = hstack([csr_matrix(X_train_select_scaled), summary_train_tfidf, text_train_tfidf])
X_test_final = hstack([csr_matrix(X_test_select_scaled), summary_test_tfidf, text_test_tfidf])

# Model Creationhsvc = LinearSVC(random_state=0, max_iter=10000, dual=False, class_weight='balanced')

# Define parameter distributions
param_distributions = {
    'C': [0.1, 1, 10],
    'loss': ['squared_hinge']
}

# Initialize RandomizedSearchCV
randomized_search = RandomizedSearchCV(
    estimator=svc,
    param_distributions=param_distributions,
    n_iter=3,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=0
)

# Perform hyperparameter tuning
randomized_search.fit(X_train_final, Y_train)

# Output best parameters
print("Best parameters from RandomizedSearchCV: ", randomized_search.best_params_)

# Train final model on full training data
best_svc = randomized_search.best_estimator_
best_svc.fit(X_train_final, Y_train)

# Predict the score using the model
Y_test_predictions = best_svc.predict(X_test_final)

# Model Evaluation
print("Accuracy on testing set = ", accuracy_score(Y_test, Y_test_predictions))
print(classification_report(Y_test, Y_test_predictions))

# Plot a confusion matrix
cm = confusion_matrix(Y_test, Y_test_predictions, normalize='true')
sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f')
plt.title('Confusion Matrix of the Best LinearSVC Classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Feature Importance Analysis
# Get feature names
numerical_feature_names = features
summary_feature_names = ['summary_' + f for f in tfidf_summary.get_feature_names_out()]
text_feature_names = ['text_' + f for f in tfidf_text.get_feature_names_out()]
all_feature_names = numerical_feature_names + summary_feature_names + text_feature_names

coefficients = best_svc.coef_

# Compute the average absolute coefficient across classes
average_coefficients = np.mean(np.abs(coefficients), axis=0)

# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': average_coefficients
})

# Sort the features by importance
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# Display top 20 features
print("Top 20 features:")
print(feature_importance_df.head(100))

# Display bottom 20 features
print("\nBottom 20 features:")
print(feature_importance_df.tail(100))

# Identify low-importance words
low_importance_features = feature_importance_df[feature_importance_df['importance'] < 0.11]
low_importance_tfidf_features = low_importance_features[
    low_importance_features['feature'].str.startswith('summary_') |
    low_importance_features['feature'].str.startswith('text_')
]
low_importance_words = low_importance_tfidf_features['feature'].apply(lambda x: x.split('_', 1)[1]).tolist()

print("\nLow-importance words to consider for stop words:")
print(low_importance_words)

# Update stop words
custom_stop_words.extend(low_importance_words)
stop_words = stopwords.words('english') + custom_stop_words

# Re-preprocess text with updated stop words
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-letter characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stop words
    return text

for df in [X_train, X_test]:
    df['Summary'] = df['Summary'].apply(preprocess_text)
    df['Text'] = df['Text'].apply(preprocess_text)

# Re-vectorize with updated TF-IDF
tfidf_summary = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words=stop_words)
tfidf_text = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words=stop_words)

tfidf_summary.fit(X_train['Summary'])
tfidf_text.fit(X_train['Text'])

summary_train_tfidf = tfidf_summary.transform(X_train['Summary'])
text_train_tfidf = tfidf_text.transform(X_train['Text'])

summary_test_tfidf = tfidf_summary.transform(X_test['Summary'])
text_test_tfidf = tfidf_text.transform(X_test['Text'])

# Re-combine features
X_train_final = hstack([csr_matrix(X_train_select_scaled), summary_train_tfidf, text_train_tfidf])
X_test_final = hstack([csr_matrix(X_test_select_scaled), summary_test_tfidf, text_test_tfidf])

# Retrain and evaluate the model
best_svc.fit(X_train_final, Y_train)

Y_test_predictions = best_svc.predict(X_test_final)

print("Accuracy on testing set after updating stop words = ", accuracy_score(Y_test, Y_test_predictions))
print(classification_report(Y_test, Y_test_predictions))

# Plot confusion matrix
cm = confusion_matrix(Y_test, Y_test_predictions, normalize='true')
sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f')
plt.title('Confusion Matrix after Updating Stop Words')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
