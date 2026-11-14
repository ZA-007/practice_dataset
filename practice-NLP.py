import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords') #used for text cleaning
nltk.download('punkt') # for tokenization if needed
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


data = pd.read_csv(
    'spam.csv',
    encoding='latin-1'
)

print("Shape:", data.shape)
print(data.head()) #shows that v1 is the target variable
print(data["v2"].head()) #shows that vs contains the message
print(data.columns)
print(data.info())
print(data["Unnamed: 2"].head(20))

data = data.rename(
    columns={
        "v1": "label",
        "v2": "message"
    }
) #rename them for readability

#drop the unnamed columns
data = data.drop(
    columns=[
        "Unnamed: 2",
        "Unnamed: 3",
        "Unnamed: 4"
    ]
)

print(data.columns)

#check for imbalance
print(data['label'].value_counts())


# now we are asked to display three random reviews:
for i in data.sample(3).index:
    print("=== SAMPLE ===")
    print(data.loc[i, 'message'][:500], "\n---label:", data.loc[i, 'label'])


#step 2 clean the text
#Objective: Implement a robust clean_text() function (HTML removal, lowercasing, non-alpha removal, optional stemming/lemmatization, stopword removal)

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def clean_text(text, do_lemmatize=True, remove_digits=False): #remove digits = false was done here bceasue keeping numbers can be useful for spam datasets ,

    # Remove HTML
    text = BeautifulSoup(
        text,
        "html.parser"
    ).get_text()

    # Remove text inside brackets
    text = re.sub(
        r'\[[^\]]*\]',
        ' ',
        text
    )

    # Remove punctuation
    if remove_digits:
        text = re.sub(
            r'[^a-zA-Z\s]',
            ' ',
            text
        )
    else:
        text = re.sub(
            r'[^a-zA-Z0-9\s]',
            ' ',
            text
        )

    # Lowercase and tokenize
    tokens = text.lower().split()

    # Remove stopwords
    tokens = [
        t for t in tokens
        if t not in stop_words
    ]

    # Lemmatize
    if do_lemmatize:
        tokens = [
            lemmatizer.lemmatize(t)
            for t in tokens
        ]

    return ' '.join(tokens)

'''remove_digits=false was done instead of = true because , Numerical values were retained during preprocessing because spam messages frequently contain phone numbers, 
monetary amounts, promotional codes and deadlines. Preserving numerical information may improve the model's ability to distinguish spam from legitimate messages.'''

#test that the cleaning function worked and visually compare raw text vs cleaned text

data_sample = data.sample(2000, random_state=42).copy()
data_sample['clean'] = data_sample['message'].apply(clean_text)
print(data_sample[['message','clean']].head(3))

#step 3 full dataset preprocessing
#Objective: Apply cleaning to whole dataset


data['clean'] = data['message'].apply(clean_text) #takes every sms message, runs it through clean text and stores it in a new clean column
data.to_csv('data_clean_short_debug.csv', index=False) #saves the cleaned dataset to a CSV file, think of it as a save progress
print("Missing values:", data['clean'].isna().sum()) #checks if any cleaned message has become NAN
#if u endded up with missing values do this : data['clean'] = data['clean'].fillna('')

#step 4 feature extraction: CountVectorizer & TF-IDF. convert text into numbers so that the ml model can understand it


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

MAX_FEAT = 5000 # manageable for a class demo, maximum number of features
bow_vec = CountVectorizer(max_features=MAX_FEAT, ngram_range=(1,2)) # BOW counts how often words appear
tfidf_vec = TfidfVectorizer(max_features=MAX_FEAT, ngram_range=(1,2)) # TF-IDF measures how important words are
X_bow = bow_vec.fit_transform(data['clean']) # Convert cleaned reviews into numbers using BOW
X_tfidf = tfidf_vec.fit_transform(data['clean'])

print("BOW shape:", X_bow.shape)
print("TF-IDF shape:", X_tfidf.shape)
#this tells us that the text to number conversion worked, so we now have 5572 messages and 5000 features/words/phrases
#CountVectorizer converted the cleaned text messages into a Bag-of-Words representation containing the 5000 most frequent words and phrases

#dispaly the frist 30 features
print(bow_vec.get_feature_names_out()[:30])


# step 5 train and test split

from sklearn.model_selection import train_test_split

y = data['label'].map({'ham':0, 'spam':1}).values # which is 0 0r 1 doesnt really matter, but typcially negative/ham/no are 0 and yes/postive/fraud/ spam are 1

#split our BOW data
X_tr_bow, X_te_bow, y_tr, y_te = train_test_split(X_bow, y, test_size=0.2, random_state=42, stratify=y)
#split our TF-IDF data
X_tr_tfidf, X_te_tfidf, _, _ = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

print("Train shape:", X_tr_bow.shape, "Test shape:", X_te_bow.shape)

#the basleine basic model , i chose complment NB isntead of Multinomial NB becasue it handles imbalanced data better
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report

cnb = ComplementNB()
cnb.fit(X_tr_bow, y_tr)
cnb_pred = cnb.predict(X_te_bow)

print(
    "ComplementNB (BOW) Accuracy:",
    accuracy_score(y_te, cnb_pred)
)

print(
    classification_report(
        y_te,
        cnb_pred
    )
)

''''The dataset exhibited class imbalance (86.6% ham and 13.4% spam). Therefore, Complement Naive Bayes was selected instead of standard Multinomial Naive Bayes, 
as it is specifically designed to improve classification performance on imbalanced text datasets.'''''

#complement NB using TF-IDF (the previous one was using bOW)
cnb_tfidf = ComplementNB()

cnb_tfidf.fit(
    X_tr_tfidf,
    y_tr
)

cnb_tfidf_pred = cnb_tfidf.predict(
    X_te_tfidf
)

print(
    "ComplementNB TF-IDF Accuracy:",
    accuracy_score(
        y_te,
        cnb_tfidf_pred
    )
)

print(
    classification_report(
        y_te,
        cnb_tfidf_pred
    )
)

# ==========================================
# COMPLEMENT NAIVE BAYES RESULTS (BOW)
# ==========================================

# The Complement Naive Bayes model achieved an
# accuracy of 97.4% when trained using Bag of
# Words features.

# The model achieved a spam recall of 95%,
# meaning it correctly identified 95% of all
# spam messages in the test dataset.

# Spam precision was 87%, indicating that
# 87% of messages predicted as spam were
# actually spam.

# The spam F1-score was 0.91, showing a strong
# balance between precision and recall.

# Overall, the Bag of Words representation
# produced excellent performance and was highly
# effective at distinguishing spam from ham
# messages.

# ==========================================
# COMPLEMENT NAIVE BAYES RESULTS (TF-IDF)
# ==========================================

# The Complement Naive Bayes model achieved an
# accuracy of 96.9% using TF-IDF features.

# Spam recall remained high at 94%, meaning the
# majority of spam messages were successfully
# detected.

# Spam precision was 84%, slightly lower than
# the Bag of Words model.

# The spam F1-score was 0.89, indicating strong
# classification performance but slightly below
# the Bag of Words representation.

# TF-IDF produced very similar results to Bag
# of Words, although Bag of Words achieved
# slightly better accuracy, precision and
# F1-score.

#advanced model 2
from sklearn.linear_model import SGDClassifier

#SDG with TF-IDF
sgd = SGDClassifier(max_iter=500, tol=1e-3, class_weight="balanced", random_state=42)

sgd.fit(X_tr_tfidf, y_tr)
sgd_tfidf_pred = sgd.predict(X_te_tfidf)

print("SGD (TF-IDF) Accuracy:", accuracy_score(y_te, sgd_tfidf_pred))

print(classification_report(y_te, sgd_tfidf_pred))

#SDG for BOW
sgd = SGDClassifier(max_iter=500, tol=1e-3, class_weight="balanced", random_state=42)

sgd.fit(X_tr_bow, y_tr)
sgd_bow_pred = sgd.predict(X_te_bow)

print("SGD (BOW) Accuracy:", accuracy_score(y_te, sgd_bow_pred))

print(classification_report(y_te, sgd_bow_pred))

# ==================================================
# SGD CLASSIFIER INTERPRETATION
# ==================================================

# SGDClassifier achieved the highest overall performance
# across all models tested.

# SGD (TF-IDF) Accuracy:
# 98.48%

# SGD (BOW) Accuracy:
# 98.21%

# TF-IDF slightly outperformed Bag-of-Words,
# suggesting that weighting words by importance
# provided more useful information than simple
# word frequency counts.

# Looking specifically at Spam detection:

# Precision (Spam):
# 0.96

# Recall (Spam):
# 0.92

# F1 Score (Spam):
# 0.94

# The high precision indicates that when the model
# predicts a message as spam it is usually correct.

# The high recall indicates that the model successfully
# identifies most spam messages.

# The strong F1 score demonstrates a good balance
# between precision and recall.

# Applying class_weight="balanced" helped compensate
# for the class imbalance in the dataset
# (86.6% ham vs 13.4% spam).

#extra stuff i can do to show in the report
#confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#naive bayes
cm_cnb = confusion_matrix(y_te, cnb_tfidf_pred)

ConfusionMatrixDisplay(
    confusion_matrix=cm_cnb,
    display_labels=["ham", "spam"]
).plot()

plt.title("ComplementNB Confusion Matrix")
plt.show()

#SDG
cm_sgd = confusion_matrix(y_te, sgd_tfidf_pred)

ConfusionMatrixDisplay(
    confusion_matrix=cm_sgd,
    display_labels=["ham", "spam"]
).plot()

plt.title("SGD Confusion Matrix")
plt.show()

#word cloud
#A word cloud is a visual representation of text where:
#Larger words = appear more frequently.
#Smaller words = appear less frequently.

from wordcloud import WordCloud

spam_text = ' '.join(
    data[data.label == 'spam']['clean'].tolist()
)

ham_text = ' '.join(
    data[data.label == 'ham']['clean'].tolist()
)

wc_spam = WordCloud(
    max_words=200
).generate(spam_text)

wc_ham = WordCloud(
    max_words=200
).generate(ham_text)

# Spam Word Cloud

plt.figure(figsize=(10, 4))
plt.imshow(wc_spam)
plt.axis('off')
plt.title('Spam Words')
plt.show()

# Ham Word Cloud

plt.figure(figsize=(10, 4))
plt.imshow(wc_ham)
plt.axis('off')
plt.title('Ham Words')
plt.show()

#The spam word cloud highlights the most frequently occurring words within spam messages.
# Terms such as "free", "call", "text", "mobile", "reply", and "claim" appear prominently,
# indicating that spam messages commonly use promotional language, incentives, competitions and calls to action.
# These findings align with the characteristics of unsolicited marketing and prize-related messages.

#top spam/ham words , this is the nlp equivalent of feature importance
def show_top_coef(classifier, vectorizer, n=20):

    feat_names = vectorizer.get_feature_names_out()

    coefs = classifier.coef_[0]

    top_pos = np.argsort(coefs)[-n:]
    top_neg = np.argsort(coefs)[:n]

    print("Top Spam Features:")
    print(feat_names[top_pos])

    print("\nTop Ham Features:")
    print(feat_names[top_neg])

show_top_coef(
    sgd,
    tfidf_vec,
    n=20
)

'''''Feature coefficient analysis was performed on the SGDClassifier to identify the most influential words driving classification decisions. 
Spam messages were strongly associated with terms such as "mobile", "reply", "txt", "ringtone", "service" and web-related words including "www" and "http". 
These words suggest a strong connection to promotional offers, subscriptions and advertising activity. In contrast, legitimate messages were associated with conversational language such as "later", "home", "sorry", "gonna" and "want talk",
 reflecting normal interpersonal communication. This demonstrates that the model learned meaningful linguistic patterns that differentiate spam from non-spam messages.'''''