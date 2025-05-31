# SMS-spam-classifier


import numpy as np
import pandas as pd

dataset = pd.read_csv("mail_data.csv")
dataset
dataset.info()
dataset['Category'] = dataset['Category'].map({'spam':1,'ham':0})
dataset
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.figure(figsize=(10,6))
sns.countplot(x="Category",data=dataset)
only_spam = dataset[dataset["Category"]==1]
only_spam
diff = int((dataset.shape[0]-only_spam.shape[0])/only_spam.shape[0])
diff
for i in range(diff):
    dataset = pd.concat([dataset,only_spam])
dataset
plt.figure(figure=(10,6))
sns.countplot(x='Category',data=dataset)
dataset['word_count']=dataset['Message'].apply(lambda x: len(x.split()))
dataset
plt.figure(figure=(15,6))

plt.subplot(1,2,1)
sns.histplot(dataset[dataset['Category'] ==0].word_count,kde=True)

plt.subplot(1,2,2)
sns.histplot(dataset[dataset['Category'] ==1].word_count,color='red',kde=True)

plt.show()
def currency(data):
    currency_symbols = ['€',' $',' ¥ ','£',' ₹']
    for i in  currency_symbols:
        if i in data:
            return 1
        
    return 0

    dataset["contains_currency_symbols"]=dataset["Message"].apply(currency)
    dataset
    plt.figure(figure=(15,6))

sns.countplot(x="contains_currency_symbols",hue="Category",data=dataset)
plt.show()

def numbers(data):
    for i in data:
        if ord(i) >=48 and ord(i) <=57:
            return 1
    return 0  

    dataset["contains_numbers"]=dataset['Message'].apply(numbers)
dataset


plt.figure(figure=(15,6))

sns.countplot(x="contains_numbers",hue="Category",data=dataset)
p - plt.title('countplot for containing numbers')
p - plt.xCategory('Does SMS contains any numbers')
p - plt.yCategory('count')
p - plt.legend(Category["Ham","Spam"],loc =9)
plt.show()

import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stopwords = nltk.corpus.stopwords.words("english")
wnl=nltk.stem.WordNetLemmatizer()

corpus =[]

for sms in list(dataset.Message):
    Message=re.sub(pattern='[^a-zA-Z]',repl=' ',string=sms)
    Message=Message.lower()
    words=Message.split()
    filtered_words=[word for word in words if word not in stopwords]
    lemm_words=[wnl.lemmatize(word) for word in filtered_words]
    Message=''.join(lemm_words)
    
    corpus.append(Message)

corpus
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=500)
vectors = tfidf.fit_transform(corpus).toarray()
feature_names = tfidf.get_feature_names_out()

x = pd.DataFrame(vectors,columns=feature_names)
y = dataset['Category']

x
y

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2)

x_test
def predict_spam(sms):
    Message=re.sub(pattern='[^a-zA-Z]',repl=' ',string=sms)
    Message=Message.lower()
    words=Message.split()
    filtered_words=[word for word in words if word not in stopwords]
    lemm_words=[wnl.lemmatize(word) for word in filtered_words]
    Message=''.join(lemm_words)
    temp = tfidf.transform([message]).toarray()
    return mnb.predict(temp)

    from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()

from sklearn.metrics import accuracy_score,confusion_matrix

mnb.fit(x_train,y_train)
y_pred = mnb.predict(x_test)

accuracy_score(y_test , y_pred)
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='g')

   
