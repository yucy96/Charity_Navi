import pandas as pd
import sys
import json
import re
from operator import itemgetter

df = pd.read_csv("input.csv")
df2 = pd.read_csv("input2.csv")
df = df[['EIN', 'Category']]
df2 = df2[['EIN', 'Data']]

data = {}

category_dict = {}
overall_dict = {}

for index, row in df2.iterrows():
    #data.row = {}
    #print(row['Data']);
    text = json.loads(str(row['Data']))
    #print(text.values());
    #print(text['QUESTIONS']);
    data[row['EIN']] = {}
    if text.get('QUESTIONS', 0):
        which_category = df.loc[(df['EIN'] == row['EIN']), 'Category'].values[0]

        #print(which_category);
        #print(category_dict[which_category]);
        if which_category not in category_dict:
            category_dict[which_category] = {}

        for elem in text['QUESTIONS']:
            for word in re.split('[ ,).(#]', elem['ANSWER']):
                regex = re.compile('[^a-zA-Z]')
                word = regex.sub('', word)
                common_words = ["the","of","and","a","to","in","is","you","that","it","he","was","for","on","are","as","with","his","they","I","at","be","this","have","from","or","one","had","by","word","but","not","what","all","were","we","when","your","can","said","there","use","an","each","which","she","do","how","their","if","will","up","other","about","out","many","then","them","these","so","some","her","would","make","like","him","into","time","has","look","two","more","write","go","see","number","way"];

                if word == '' or word in common_words:
                    continue

                word = word.lower()
                word = word.encode('ascii',errors='ignore').decode()

                tmp = data[row['EIN']].get(word, 0)
                count = category_dict[which_category].get(word, 0)
                total_count = overall_dict.get(word, 0)
                data[row['EIN']][word] = tmp + 1
                category_dict[which_category][word] = count + 1
                overall_dict[word] = total_count + 1
'''
for key in category_dict:
    f = open("wordclouds/" + key, 'w');
    d = category_dict[key]
    for k, v in sorted(d.items(), key=itemgetter(1), reverse=True):
        f.write("%s: %d\n" % (k, v))

'''
f = open("all_words.txt", 'w')
for k, v in sorted(overall_dict.items(), key=itemgetter(1), reverse=True):
    f.write("%s: %d\n" % (k, v))
    for key in category_dict:
        d = category_dict[key]
        '''
        for k2, v2 in sorted(d.items(), key=itemgetter(1), reverse=True):
            f.write("%s: %d\n" % (k2, v2))
        '''
        key2 = k
        val2 = category_dict[key].get(key2, 0)
        f.write("\t%s: %d\n" % (key,val2))
    f.write("\n")
f.close()
