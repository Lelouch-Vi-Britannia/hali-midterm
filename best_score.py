# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from os.path import exists
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier
from scipy.sparse import hstack, csr_matrix
import seaborn as sns

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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

    # Convert Time to datetime
    df['ReviewTime'] = pd.to_datetime(df['Time'], unit='s', errors='coerce')
    df['Year'] = df['ReviewTime'].dt.year.fillna(0).astype(int)
    df['Month'] = df['ReviewTime'].dt.month.fillna(0).astype(int)
    df['DayOfWeek'] = df['ReviewTime'].dt.dayofweek.fillna(0).astype(int)

    # Text length features
    df['Summary_length'] = df['Summary'].fillna('').apply(len)
    df['Text_length'] = df['Text'].fillna('').apply(len)

    return df


# Apply feature extraction
trainingSet = add_features_to(trainingSet)

# Merge testingSet with trainingSet to get the features for the test set
X_submission = pd.merge(testingSet, trainingSet.drop(columns=['Score']), on='Id', how='left')

# Handle missing values in X_submission
X_submission.fillna(
    {'Summary': '', 'Text': '', 'HelpfulnessNumerator': 0, 'HelpfulnessDenominator': 0, 'Helpfulness': 0,
     'Year': 0, 'Month': 0, 'DayOfWeek': 0, 'Summary_length': 0, 'Text_length': 0}, inplace=True)

# Compute user and product statistics from training data
user_stats = trainingSet.groupby('UserId')['Score'].agg(['mean', 'std']).reset_index()
user_stats = user_stats.rename(columns={'mean': 'User_avg_score', 'std': 'User_std_score'})

product_stats = trainingSet.groupby('ProductId')['Score'].agg(['mean', 'std']).reset_index()
product_stats = product_stats.rename(columns={'mean': 'Product_avg_score', 'std': 'Product_std_score'})

# Merge user and product stats into trainingSet and X_submission
trainingSet = pd.merge(trainingSet, user_stats, on='UserId', how='left')
trainingSet = pd.merge(trainingSet, product_stats, on='ProductId', how='left')

X_submission = pd.merge(X_submission, user_stats, on='UserId', how='left')
X_submission = pd.merge(X_submission, product_stats, on='ProductId', how='left')

# Handle missing values in stats
global_mean_score = trainingSet['Score'].mean()

for df in [trainingSet, X_submission]:
    df['User_avg_score'] = df['User_avg_score'].fillna(global_mean_score)
    df['User_std_score'] = df['User_std_score'].fillna(0)
    df['Product_avg_score'] = df['Product_avg_score'].fillna(global_mean_score)
    df['Product_std_score'] = df['Product_std_score'].fillna(0)

# Prepare Feature Set and Labels
features = [
    'HelpfulnessNumerator',
    'HelpfulnessDenominator',
    'Helpfulness',
    'Year',
    'Month',
    'DayOfWeek',
    'Summary_length',
    'Text_length',
    'User_avg_score',
    'User_std_score',
    'Product_avg_score',
    'Product_std_score'
]

# Add sentiment features
sia = SentimentIntensityAnalyzer()


def get_sentiment_scores(text):
    if pd.isna(text):
        text = ''
    return sia.polarity_scores(text)['compound']


for df in [trainingSet, X_submission]:
    df['Text_sentiment'] = df['Text'].apply(get_sentiment_scores)
    df['Summary_sentiment'] = df['Summary'].apply(get_sentiment_scores)

features.extend(['Text_sentiment', 'Summary_sentiment'])

# Remove identifiers from features
trainingSet = trainingSet.drop(columns=['UserId', 'ProductId', 'ReviewTime', 'Time'])
X_submission = X_submission.drop(columns=['UserId', 'ProductId', 'ReviewTime', 'Time'])

X_train = trainingSet[trainingSet['Score'].notnull()]
Y_train = X_train['Score']

# Split a test set from the training data
X_train_split, X_test_split, Y_train_split, Y_test_split = train_test_split(
    X_train,
    Y_train,
    test_size=0.25,
    random_state=0,
    stratify=Y_train
)

# Text Preprocessing
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


for df in [X_train_split, X_test_split, X_submission]:
    df['Summary'] = df['Summary'].apply(preprocess_text)
    df['Text'] = df['Text'].apply(preprocess_text)

# After preprocessing, select the features
X_train_select = X_train_split[features]
X_test_select = X_test_split[features]
X_submission_select = X_submission[features]

# Scale numerical features
scaler = StandardScaler()
X_train_select_scaled = scaler.fit_transform(X_train_select)
X_test_select_scaled = scaler.transform(X_test_select)
X_submission_select_scaled = scaler.transform(X_submission_select)

# Feature Engineering: Text Features
tfidf_summary = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words='english')
tfidf_text = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')

# Fit and transform the text data
tfidf_summary.fit(X_train_split['Summary'])
tfidf_text.fit(X_train_split['Text'])

summary_train_tfidf = tfidf_summary.transform(X_train_split['Summary'])
text_train_tfidf = tfidf_text.transform(X_train_split['Text'])

summary_test_tfidf = tfidf_summary.transform(X_test_split['Summary'])
text_test_tfidf = tfidf_text.transform(X_test_split['Text'])

summary_submission_tfidf = tfidf_summary.transform(X_submission['Summary'])
text_submission_tfidf = tfidf_text.transform(X_submission['Text'])

# Combine numerical and text features
X_train_combined = hstack([csr_matrix(X_train_select_scaled), summary_train_tfidf, text_train_tfidf])
X_test_combined = hstack([csr_matrix(X_test_select_scaled), summary_test_tfidf, text_test_tfidf])
X_submission_combined = hstack(
    [csr_matrix(X_submission_select_scaled), summary_submission_tfidf, text_submission_tfidf])

# Handle class imbalance with class_weight
svc = LinearSVC(random_state=0, max_iter=10000, dual=False, class_weight='balanced')

# Hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
    'loss': ['squared_hinge']
}

grid_search = GridSearchCV(
    estimator=svc,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

# Perform hyperparameter tuning on a sample to save time
sample_size = 50000
X_train_sample, _, Y_train_sample, _ = train_test_split(
    X_train_combined,
    Y_train_split,
    train_size=sample_size,
    random_state=0,
    stratify=Y_train_split
)

grid_search.fit(X_train_sample, Y_train_sample)
best_svc = grid_search.best_estimator_
print("Best parameters from GridSearchCV: ", grid_search.best_params_)

# Train final model on full training data
best_svc.fit(X_train_combined, Y_train_split)

# Predict the score using the model
Y_test_predictions = best_svc.predict(X_test_combined)

# Model Evaluation
print("Accuracy on testing set = ", accuracy_score(Y_test_split, Y_test_predictions))
print(classification_report(Y_test_split, Y_test_predictions))

# Plot a confusion matrix
cm = confusion_matrix(Y_test_split, Y_test_predictions, normalize='true')
sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f')
plt.title('Confusion Matrix of the Best LinearSVC Classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Create submission file
X_submission['Score'] = best_svc.predict(X_submission_combined)
submission = X_submission[['Id', 'Score']]
submission.to_csv("./data/submission_combined.csv", index=False)
