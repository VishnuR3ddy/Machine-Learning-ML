**🎬 Sentiment Analysis on Movie Reviews**

This project performs sentiment analysis on movie reviews using Natural Language Processing (NLP) techniques. The goal is to classify reviews as either positive or negative based on the content.

**📜 Table of Contents**

Overview
Dataset
Exploratory Data Analysis (EDA)
Preprocessing
Modeling
Model Evaluation
How to Use
Results
Conclusion

**📘 Overview**

This project implements a Naive Bayes Classifier to predict the sentiment of movie reviews. It includes:

Data preprocessing (removing stop words, stemming, etc.)
Feature extraction using TfidfVectorizer
Model training and evaluation
Saving and loading models for future use

**📊 Dataset**

The dataset used is an IMDB movie reviews dataset with two columns:

Review: The text of the movie review.
Sentiment: The sentiment associated with the review, labeled as Positive or Negative.
The dataset can be found in IMDB.csv.

**🔍 Exploratory Data Analysis (EDA)**

We perform EDA to understand the dataset:

Shape of the dataset
Null values check
Distribution of sentiment labels
Visualization libraries like Seaborn are used to create count plots to show the distribution of positive and negative reviews.

**🛠️ Preprocessing
**
Several preprocessing steps are applied to the raw text data:

Text cleaning: Remove special characters and numbers.
Lowercasing: Convert all words to lowercase.
Tokenization: Split sentences into individual words.
Stopwords removal: Eliminate common but irrelevant words.
Stemming: Reduce words to their base form using PorterStemmer.
These steps create a cleaned corpus ready for feature extraction.

**📐 Modeling**

The features from the cleaned text data are extracted using TfidfVectorizer with a maximum feature limit of 5000. The target label is encoded as:

Positive: 1
Negative: 0
We split the dataset into a training set and a test set with an 80/20 ratio and train the model using Multinomial Naive Bayes.

**🧪 Model Evaluation**

The model is evaluated using:

Accuracy score
Confusion matrix
Classification report (precision, recall, F1-score)

**💻 How to Use**

Clone this repository:
bash
Copy code
git clone https://github.com/yourusername/movie-review-sentiment-analysis.git
cd movie-review-sentiment-analysis
Install the required libraries:
bash
Copy code
pip install -r requirements.txt
To train the model, run the notebook or execute:
bash
Copy code
python train.py
To use the trained model for sentiment prediction:
python
Copy code
def test_model(sentence):
    sen = save_cv.transform([sentence]).toarray()
    res = model.predict(sen)[0]
    if res == 1:
        return 'POSITIVE review'
    else:
        return 'NEGATIVE review'
**Example**
python
Copy code
sentence = "This is a fantastic movie!"
print(test_model(sentence))  # Expected: POSITIVE review

**📈 Results**

The final model achieved the following performance metrics on the test set:

Accuracy: 85%
Precision, Recall, F1-Score (as seen in the classification report)

**🚀 Conclusion**

This project demonstrates the use of Naive Bayes for text classification in NLP. The trained model performs well on movie reviews, achieving an accuracy of around 85%. It can be extended to other types of textual data by modifying the preprocessing and training steps.

**🗂️ Files in the Repository**

SENTIMENT_ANALYSIS_ON_MOVIE_REVIEWS.ipynb: Main Jupyter notebook.
IMDB.csv: Movie reviews dataset.
count-Vectorizer.pkl: Saved TfidfVectorizer model.
Movies_Review_Classification.pkl: Trained Naive Bayes classifier model.
train.py: Python script to train the model.

