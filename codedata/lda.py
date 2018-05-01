import matplotlib
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import warnings
#warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import re, nltk, spacy, gensim
import sys
import json
from operator import itemgetter

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
lemmatizer = WordNetLemmatizer()
vectorizer = CountVectorizer(analyzer='word',
                             min_df=10,  # minimum reqd occurences of a word
                             stop_words='english',  # remove stop words
                             lowercase=True,  # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                             )
lda_model = LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
             evaluate_every=-1, learning_decay=0.3, learning_method=None,
             learning_offset=10.0, max_doc_update_iter=100, max_iter=10,
             mean_change_tol=0.001, n_components=10, n_jobs=1, n_topics=15,
             perp_tol=0.1, random_state=None, topic_word_prior=None,
             total_samples=1000000.0, verbose=0)
df = pd.read_csv("input.csv")
df2 = pd.read_csv("input2.csv")
df = df[['EIN', 'Category']]
df2 = df2[['EIN','Source', 'Data']]

data = {}

category_dict = {}
overall_dict = {}

# Tokenize and Clean-up using gensim’s simple_preprocess()
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
# --------------------------------------------------------------
# Lemmatization
def lemmatize(tokens):
    stemmed = []
    for word,tag in tokens:
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a','r','n','v'] else None
        if not wntag:
            lemma = word
        else:
            lemma = lemmatizer.lemmatize(word,wntag)
        stemmed.append(lemma)
    lemmas = ' '.join(word for word in stemmed)
    return lemmas
# -------------------------------------------------------------------------
# Find top 15 key words
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    article = []
    docs = []
    for index, row in df2.iterrows():
        #data.row = {}
        text = json.loads(str(row['Data']))
        source = row['Source']
        if source != 'GuideStar':
            continue
        #print(text.values());
        temp = []
        data_lemmatized = []

        for i in range(np.size(text['QUESTIONS'])):
            temp.append(text['QUESTIONS'][i]['ANSWER'])
        data_words = list(sent_to_words(temp))
        if data_words:
            docs.append(row['EIN'])
            for item in data_words:
                data_lemmatized.append(lemmatize(pos_tag(item)))
            article.append(' '.join(word for word in data_lemmatized))
        else:
            continue
    data_vectorized = vectorizer.fit_transform(article)

# Run LDA-------------------------------------------------------------------------------------
    # Create Document - Topic Matrix
    lda_output = lda_model.fit_transform(data_vectorized)

    # column names
    topicnames = ["Topic" + str(i) for i in range(lda_model.n_topics)]

    # index names
    docnames = docs

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic

    topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=15)

    # Topic - Keywords Dataframe
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]

    df_document_topic.to_csv('doc_topics3.csv')
    df_topic_keywords.to_csv('topic_keywords3.csv')

    # Perplexity
    print("Perplexity: ", lda_model.perplexity(data_vectorized))
    print("Log Likelihood: ", lda_model.score(data_vectorized))