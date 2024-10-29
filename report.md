# First Sight
One of the key challenges of this task is the inconsistency in user ratings. Numerous factors—such as mood, personal experiences, and even random external events—can significantly influence how users rate a movie, making it difficult for the model to achieve very high accuracy in its predictions.

# Data Analysis
Upon analyzing the provided dataset, I found it to be highly imbalanced. Over 50% of the reviews have a 5-star rating, around 20% have a 4-star rating, approximately 10% have a 3-star rating, and 1-star and 2-star ratings each account for roughly 5% of the dataset. This imbalance poses a significant challenge, as the model may become biased toward predicting the majority class. To address this issue, techniques such as oversampling, undersampling, or weighting classes may be necessary to improve model performance across all rating categories.

Additionally, I identified missing values in both the summary and text columns—32 missing values in the summary and 62 in the text. Given the large size of the dataset (1,697,533 reviews), these missing values represent a very small proportion, so I opted to drop the rows containing them to maintain data consistency. I also noticed that some reviews had zero values in the 'HelpfulnessNumerator' and 'HelpfulnessDenominator' columns. Since not all reviews receive helpfulness ratings, this is expected. To avoid division by zero, I added 1 to all denominators during feature engineering.

# Feature Engineering
The initial feature engineering was guided by intuition and further refined through analysis of the results and statistical insights. The following features were extracted or engineered to improve model performance:

## Year
As mentioned in class by one of the presenters, ratings change gradually over time, which can impact the results. This makes sense, as human rating behavior evolves over time. However, my intuition suggests that this feature may have only a minor impact on model performance. Additionally, tracking this trend can be challenging, and the model's performance could degrade over the years if the trend shifts. Therefore, this feature may ultimately be excluded from the model.

## Month
I believe that the month in which a movie is reviewed could have a more significant impact on its rating. Different types of movies tend to be released during specific periods—e.g., more family-oriented films are often released during holidays. This seasonality could influence how different demographic groups rate movies, resulting in variability in ratings that the model could capture.

## Day of the Week
The "Day of the Week" feature could provide insights into user behavior patterns. For instance, people may be more inclined to leave reviews on weekends when they have more free time. Such temporal trends in user engagement might offer useful signals that the model could leverage to better understand rating behavior.

Overall, while there are valid arguments for including time-based features, I would lean towards excluding them if they do not significantly enhance the model's performance. Keeping the model streamlined is crucial to ensure that added complexity is justified by tangible improvements in predictive power.

## Usefulness Score
I believe this feature will have a significant impact on the prediction outcomes, especially for users who write a large number of reviews and receive many usefulness scores. For example, a person with a high usefulness score may be perceived as a rational reviewer and is less likely to give extreme ratings (e.g., 1 or 5) without a valid reason. If such a reviewer uses moderate language in their review, they are more likely to provide a rating between 2 and 4. This suggests that usefulness scores can serve as an indicator of rating behavior, helping the model predict more accurate and consistent ratings.

## Summary and Review Text
The summary and review text are undoubtedly the most important features, as they provide the most information about the user's attitude towards a movie. The major challenge is to effectively reduce the dimensionality of this information—extracting the key insights from the entire paragraph while retaining the essential meaning. This requires careful preprocessing and feature extraction techniques, such as TF-IDF or sentiment analysis, to capture the critical aspects of user opinions while managing the complexity of the text data.

## Average and Standard Deviation of Score for Movie and User
These metrics were among the most important features in the model as they capture the general expectation and variability of how a movie is rated, as well as the rating patterns of individual users. By including these features, the model can better understand the typical rating behavior for a particular movie and identify if a user's rating deviates from the norm. Through further experimentation, I discovered that achieving similar accuracy without these metrics was very challenging, highlighting their critical role in improving predictive performance.

# Algorithms
The best-performing model includes all the features mentioned above, along with the lengths of the summary and review text. For text data, I applied **TF-IDF vectorization** with a maximum of 2000 features for the summary and 5000 features for the review text to transform them into numerical form suitable for the model. I set the **n-gram range to (1, 2)** to capture both unigrams and bigrams—allowing the model to understand phrases like "not good" or "not bad," where the meaning changes significantly when words are combined rather than considered individually. Numerical features were standardized using **`StandardScaler`** to ensure consistency in feature values across different scales, making the model more stable and effective.

For the final prediction, I employed a **Linear Support Vector Classifier (LinearSVC)**. To optimize the model, I used **RandomizedSearchCV** for hyperparameter tuning. The search involved testing different regularization parameter (`C`) values: `[0.1, 1, 10]`, with a **squared hinge** loss function, and applying **3-fold cross-validation**. This process allowed for efficient tuning to achieve the best model configuration. The combination of comprehensive feature engineering, effective text transformation, and careful hyperparameter optimization helped improve the predictive performance of the model.

# Citations and Justifications

**TF-IDF**: Recommended by GPT. **Term Frequency (TF)** measures how often a word appears in a document, and the **Inverse Document Frequency (IDF)** down-weights terms that appear frequently across many documents. Together, TF-IDF highlights the important words across the entire corpus. This method aims to emphasize key terms that contribute meaningfully to the content of the text, ultimately improving the model's predictive capabilities.

**RandomizedSearchCV**: Recommended by GPT to replace **GridSearchCV**. **RandomizedSearchCV** is a hyperparameter optimization technique that searches for the best parameters by sampling from a specified distribution. Compared to **GridSearchCV**, it is more efficient for large search spaces as it does not exhaustively try all possible combinations, making it suitable for scenarios with limited computational resources.

**LinearSVC**: I learned this from CS 542 (Machine Learning). I chose **LinearSVC** because it is generally more robust to noise and can achieve better generalization on unseen data, especially when dealing with high-dimensional data. Unlike **k-Nearest Neighbors (kNN)**, which suffers greatly from the curse of dimensionality with thousands of features, LinearSVC can handle such complexity more effectively. Additionally, it is computationally efficient since kNN requires recalculating distances for every fold, which is time-consuming for a large dataset (1.4GB). LinearSVC's time efficiency also made it preferable to **Random Forest**, which GPT recommended, but would have required more time to train.

# Findings

## Overfitting
The highest score I achieved was 68.5% accuracy locally, but it dropped to 66.4% on Kaggle. This discrepancy suggests overfitting, confirming my intuition during feature engineering that certain features, such as time, could be removed to maintain model robustness. Overfitting likely occurred because the model captured patterns specific to the training set that did not generalize well to new data. Future iterations should focus on simplifying the model and removing less impactful features to improve generalization.

## Limitations of TF-IDF
Contrary to my expectations, TF-IDF did not consistently highlight the words most crucial for classification. While some top-rated words like "great," "fantastic," and "terrible" were useful in indicating sentiment, other frequent words like "movie" and "film" added noise to the model as they are generic nouns. This outcome is understandable, given that the dataset consists of movie reviews, and such generic terms do not meaningfully differentiate between sentiments. Addressing this issue will be a priority in future iterations, focusing on techniques to reduce the impact of irrelevant terms, such as better stopword filtering or using more sophisticated feature selection.

## Feature Importance
In subsequent experiments, I only used feature: helpfulness, summary length, text length, and TF-IDF-processed text, as there was a sign of overfitting and I initially misunderstood the guidelines regarding the use of score-related features.

To determine feature importance, I followed GPT's suggestion to generate feature importance scores for each feature. This involved training the model with the complete set of TF-IDF features, calculating the average absolute value of the coefficients from the trained **LinearSVC**, and mapping these coefficients to their respective features to rank their importance. Interestingly, this method effectively distinguished emotionally charged words from generic ones.

The importance scores for the 3000 features ranged from 0.015 to 1.13, with only about 500 of them scoring above 0.5. This suggests that the model relies heavily on a relatively small subset of features. However, when I set the importance score threshold to 0.5, the model's performance dropped by 2%, indicating that other features still contributed meaningfully to the predictions. Ultimately, I found a sweet spot with a threshold of 0.1, which eliminated 50% of the features without significantly impacting performance.

Interestingly, some words like "perfect" did not receive high importance scores, likely because they appeared less frequently in the dataset. This reflects the imbalance in dataset—a word like "perfect" might intuitively seem highly positive, but its limited usage reduced its influence. Another example would be "good" which ranked behind 500. This reflects the inherent inconsistency in human reviews that I previously mentioned—a word like "good" can be used across different ratings, from 3 to 5 stars, and even occasionally for 1 or 2 stars, depending on the reviewer’s perspective. On the other hand, more commonly used words like "great" conveyed a consistently positive sentiment.

The refined model achieved a score of 6.39 locally and 6.43 on the Kaggle submission, suggesting that it is less vulnerable to overfitting. The model appears to have effectively captured key information from the text, even without relying on score-related features.

# Future Work and Conclusion
One major direction for future work is to combine multiple models to explore how enhanced feature engineering, including metrics like average score, might improve performance. Additionally, I observed that certain words like "perfect" were not highly rated in terms of importance, possibly due to their lower frequency. It would be interesting to explore further feature engineering methods to increase the importance of such terms, as they are intuitively meaningful and likely correlated with positive ratings.

Another potential improvement is to balance the dataset by sampling evenly across different rating classes. This could help improve model performance for underrepresented classes without significantly compromising accuracy for the dominant class (i.e., rating 5). I did not pursue this approach earlier because I believed that compromising the accuracy of class 5 predictions could be too risky, given that the dataset is naturally biased towards high ratings.

Overall, this project provided valuable insights into feature engineering and the challenges of text classification, especially with inherently noisy data like user reviews. While the current model performed reasonably well, there is ample opportunity for further refinement—particularly in improving feature selection, addressing overfitting, and experimenting with balanced datasets. Moving forward, my focus will be on building a more generalized model that captures the nuances of user sentiment more effectively while avoiding overfitting.
