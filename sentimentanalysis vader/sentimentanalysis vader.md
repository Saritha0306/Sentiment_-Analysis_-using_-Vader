```python
import pandas as pd
```


```python
#importing the training data
imdb_data=pd.read_csv('IMDB dataset.csv')
print(imdb_data.shape)
imdb_data.head(10)
```

    (50000, 2)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>One of the other reviewers has mentioned that ...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A wonderful little production. &lt;br /&gt;&lt;br /&gt;The...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I thought this was a wonderful way to spend ti...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Basically there's a family where a little boy ...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Petter Mattei's "Love in the Time of Money" is...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Probably my all-time favorite movie, a story o...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>6</th>
      <td>I sure would like to see a resurrection of a u...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>7</th>
      <td>This show was an amazing, fresh &amp; innovative i...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Encouraged by the positive comments about this...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>9</th>
      <td>If you like original gut wrenching laughter yo...</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Summary of the dataset
imdb_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>50000</td>
      <td>50000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>49582</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Loved today's show!!! It was a variety and not...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>5</td>
      <td>25000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#sentiment count
imdb_data['sentiment'].value_counts()
```




    positive    25000
    negative    25000
    Name: sentiment, dtype: int64




```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
from sklearn.metrics import accuracy_score, classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the IMDb dataset from a CSV file
df = pd.read_csv('IMDB Dataset.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Create a CountVectorizer or TfidfVectorizer to convert text data into numerical features
vectorizer = CountVectorizer(max_df=0.85, min_df=0.01, stop_words='english')
# vectorizer = TfidfVectorizer(max_df=0.85, min_df=0.01, stop_words='english')  # You can also use TF-IDF

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the number of estimators
rf_classifier.fit(X_train_vectorized, y_train)
y_pred_rf = rf_classifier.predict(X_test_vectorized)

# Evaluate the Random Forest Classifier
print("Random Forest Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Define a function for sentiment analysis using TextBlob
def analyze_sentiment_with_textblob(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "positive"
    elif sentiment < 0:
        return "negative"
    else:
        return "neutral"

# Apply TextBlob sentiment analysis to the IMDb dataset
y_pred_textblob = [analyze_sentiment_with_textblob(text) for text in X_test]

# Evaluate TextBlob-based sentiment analysis
print("\nTextBlob Sentiment Analysis Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_textblob))
print("Classification Report:")
print(classification_report(y_test, y_pred_textblob))

# Initialize VADER SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Calculate VADER sentiment scores and classify
df['VADER_Sentiment'] = df['sentiment'].apply(lambda x: analyzer.polarity_scores(x))
threshold = 0.1  # Define a threshold for VADER sentiment classification
df['VADER_Label'] = df['VADER_Sentiment'].apply(lambda x: 'positive' if x['compound'] >= threshold else 'negative' if x['compound'] <= -threshold else 'neutral')

# Evaluate VADER-based sentiment analysis
y_test_vader = df.loc[y_test.index]['VADER_Label']
print("\nVADER Sentiment Analysis Results:")
print("Accuracy:", accuracy_score(y_test, y_test_vader))
print("Classification Report:")
print(classification_report(y_test, y_test_vader))

```

    Random Forest Classifier Results:
    Accuracy: 0.8348
    Classification Report:
                  precision    recall  f1-score   support
    
        negative       0.83      0.84      0.83      4961
        positive       0.84      0.83      0.83      5039
    
        accuracy                           0.83     10000
       macro avg       0.83      0.83      0.83     10000
    weighted avg       0.83      0.83      0.83     10000
    
    
    TextBlob Sentiment Analysis Results:
    Accuracy: 0.6924
    Classification Report:
    

    C:\anaconda\Lib\site-packages\sklearn\metrics\_classification.py:1469: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\anaconda\Lib\site-packages\sklearn\metrics\_classification.py:1469: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\anaconda\Lib\site-packages\sklearn\metrics\_classification.py:1469: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

                  precision    recall  f1-score   support
    
        negative       0.89      0.43      0.58      4961
         neutral       0.00      0.00      0.00         0
        positive       0.63      0.95      0.76      5039
    
        accuracy                           0.69     10000
       macro avg       0.51      0.46      0.45     10000
    weighted avg       0.76      0.69      0.67     10000
    
    
    VADER Sentiment Analysis Results:
    Accuracy: 1.0
    Classification Report:
                  precision    recall  f1-score   support
    
        negative       1.00      1.00      1.00      4961
        positive       1.00      1.00      1.00      5039
    
        accuracy                           1.00     10000
       macro avg       1.00      1.00      1.00     10000
    weighted avg       1.00      1.00      1.00     10000
    
    


```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the IMDb dataset from a CSV file
df = pd.read_csv('IMDB Dataset.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Create a CountVectorizer to convert text data into numerical features
vectorizer = CountVectorizer(max_df=0.85, min_df=0.01, stop_words='english')

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_vectorized, y_train)
y_pred_rf = rf_classifier.predict(X_test_vectorized)

# Calculate Random Forest accuracy
rf_accuracy = accuracy_score(y_test, y_pred_rf)

# Define a function for sentiment analysis using TextBlob
def analyze_sentiment_with_textblob(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "positive"
    elif sentiment < 0:
        return "negative"
    else:
        return "neutral"

# Apply TextBlob sentiment analysis to the IMDb dataset
y_pred_textblob = [analyze_sentiment_with_textblob(text) for text in X_test]

# Calculate TextBlob accuracy
textblob_accuracy = accuracy_score(y_test, y_pred_textblob)

# Initialize VADER SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Calculate VADER sentiment scores and classify
df['VADER_Sentiment'] = df['sentiment'].apply(lambda x: analyzer.polarity_scores(x))
threshold = 0.1  # Define a threshold for VADER sentiment classification
df['VADER_Label'] = df['VADER_Sentiment'].apply(lambda x: 'positive' if x['compound'] >= threshold else 'negative' if x['compound'] <= -threshold else 'neutral')

# Calculate VADER accuracy
y_test_vader = df.loc[y_test.index]['VADER_Label']
vader_accuracy = accuracy_score(y_test, y_test_vader)

# Create a bar plot to compare accuracy scores
methods = ['Random Forest', 'TextBlob', 'VADER']
accuracy_scores = [rf_accuracy, textblob_accuracy, vader_accuracy]

plt.bar(methods, accuracy_scores, color=['blue', 'green', 'red'])
plt.xlabel('Sentiment Analysis Method')
plt.ylabel('Accuracy Score')
plt.title('Comparison of Sentiment Analysis Methods')
plt.ylim(0, 1.0)

# Show the bar plot
plt.show()

```


    
![png](output_5_0.png)
    



```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix for Random Forest Classifier
confusion_rf = confusion_matrix(y_test, y_pred_rf)

# Plot the confusion matrix for Random Forest Classifier
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest Classifier')
plt.show()

# Calculate the confusion matrix for TextBlob-based sentiment analysis
confusion_textblob = confusion_matrix(y_test, y_pred_textblob)

# Plot the confusion matrix for TextBlob-based sentiment analysis
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_textblob, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - TextBlob-based Sentiment Analysis')
plt.show()

# Calculate the confusion matrix for VADER-based sentiment analysis
confusion_vader = confusion_matrix(y_test, y_test_vader)

# Plot the confusion matrix for VADER-based sentiment analysis
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_vader, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - VADER-based Sentiment Analysis')
plt.show()

```


    
![png](output_6_0.png)
    



    
![png](output_6_1.png)
    



    
![png](output_6_2.png)
    



```python

```
