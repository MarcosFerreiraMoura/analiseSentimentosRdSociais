import numpy as np 
import pandas as pd 
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score
import pickle


data = pd.read_csv(r'data\imdb-reviews-pt-br.csv')
#data = data[['review']]

print(data.shape)
data.head()

data.info()

#print(data.text_en[0])

def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned,'',text)
data.text_en = data.text_en.apply(clean)

#print('\n\n Remove tags HTMLs\n' + reviewCleaned)

#remove os caracteres especiais 
def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem =rem+i
        else:
            rem = rem + ' '
    return rem
data.text_en = data.text_en.apply(is_special)
#print('\n\n Remove caracteres especiais\n' + semCharEspeciais )

#convertendo para lowCase
def to_lower(text):
    return text.lower()
data.text_en = data.text_en.apply(to_lower)
#print('\n\n coloca texto em minusculo\n'+ lowCase)


#remover palavras inuteis para a analise(o, a, em, no, na, do, de, e ...)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return[w for w in words if w not in stop_words]
data.text_en = data.text_en.apply(rem_stopwords)
#print('\n\n Remover Artigos\n'+' '.join(limpaArtTxt))

#
def stem_txt(text):
    ss = SnowballStemmer('english')
    return " " .join([ss.stem(w) for w in text])

data.text_en = data.text_en.apply(stem_txt)

#print('\n\n tratar texto, reduzir palavras\n')

#Maquina preditiva
#criando uma bag of words


#x = np.array(data.iloc[:,0].values)
y = np.array(data.sentiment.values)
cv = CountVectorizer(max_features= 1000)
x = cv.fit_transform(data.text_en).toarray()


print("X shape = ", x.shape)
print("Y shape = ", y.shape)
#print(y)
trainx, testx, trainy, testy = train_test_split(x, y, test_size= 0.2,random_state= 9 )
print("Train shapes: X = {}, y ={} ".format(trainx.shape, trainy.shape))
print("Test shapes: X = {}, y ={} ".format(testx.shape, testy.shape))

#Definindo modelos e treinando eles

gnb, mnb, bnb = GaussianNB(), MultinomialNB(alpha=1.0, fit_prior=True), BernoulliNB(alpha=1.0, fit_prior=True)

gnb.fit(trainx, trainy)
mnb.fit(trainx, trainy)
bnb.fit(trainx, trainy)

#predicao e acuracia para escolher o melhor modelo

ypg = gnb.predict(testx)
ypm = mnb.predict(testx)
ypb = bnb.predict(testx)

print("Gaussian = ",accuracy_score(testy, ypg))
print("Multinominal= ",accuracy_score(testy, ypm))
print("Bernoulli = ",accuracy_score(testy, ypb))

#salva o modelo treinado
pickle.dump(bnb, open('model0.pk1','wb'))
# Salvar o CountVectorizer
pickle.dump(cv, open('cv.pk1', 'wb'))


