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
    # Define Search Param
    search_params = {'n_topics': [10, 12, 15, 20, 25, 30], 'learning_decay': [.3, .5, .7, .9]}
    # Init the Model
    lda = LatentDirichletAllocation()
    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params)
    # Do the Grid Search
    model.fit(data_vectorized)
    # Best Model
    best_lda_model = model.best_estimator_
    # Model Parameters
    print("Best Model's Params: ", model.best_params_)
    # Log Likelihood Score
    print("Best Log Likelihood Score: ", model.best_score_)
    # Perplexity
    print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))
    # Get Log Likelyhoods from Grid Search Output
    n_topics = [10, 12, 15, 20, 25, 30]
    log_likelyhoods_3 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if
                         gscore.parameters['learning_decay'] == 0.3]
    log_likelyhoods_5 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if
                         gscore.parameters['learning_decay'] == 0.5]
    log_likelyhoods_7 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if
                         gscore.parameters['learning_decay'] == 0.7]
    log_likelyhoods_9 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if
                         gscore.parameters['learning_decay'] == 0.9]
    # Show graph
    plt.figure(figsize=(12, 8))
    plt.plot(n_topics, log_likelyhoods_3, label='0.3')
    plt.plot(n_topics, log_likelyhoods_5, label='0.5')
    plt.plot(n_topics, log_likelyhoods_7, label='0.7')
    plt.plot(n_topics, log_likelyhoods_9, label='0.9')
    plt.title("Choosing Optimal LDA Model")
    plt.xlabel("Num Topics")
    plt.ylabel("Log Likelyhood Scores")
    plt.legend(title='Learning decay', loc='best')
    plt.show()

    # Create Document - Topic Matrix
    lda_output = best_lda_model.transform(data_vectorized)

    # column names
    topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_topics)]

    # index names
    docnames = docs

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic


    # Styling
    def color_green(val):
        color = 'green' if val > .1 else 'black'
        return 'color: {col}'.format(col=color)


    def make_bold(val):
        weight = 700 if val > .1 else 400
        return 'font-weight: {weight}'.format(weight=weight)


    # Apply Style
    df_document_topics = df_document_topic.style.applymap(color_green).applymap(make_bold)