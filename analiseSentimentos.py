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

def carregar_dados():
    data = pd.read_csv(r'data\imdb-reviews-pt-br.csv', encoding='utf-8')
    return data
#data = data[['review']]

def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned,'',text)
#carregar_dados().text_pt = carregar_dados().text_pt.apply(clean)

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
#carregar_dados().text_pt = carregar_dados().text_pt.apply(is_special)
#print('\n\n Remove caracteres especiais\n' + semCharEspeciais )

#convertendo para lowCase
def to_lower(text):
    return text.lower()
#carregar_dados().text_pt = carregar_dados().text_pt.apply(to_lower)
#print('\n\n coloca texto em minusculo\n'+ lowCase)


#remover palavras inuteis para a analise(o, a, em, no, na, do, de, e ...)
nltk.download('stopwords')
nltk.download('punkt')

def rem_stopwords(text):
    stop_words = set(stopwords.words('portuguese'))
    words = word_tokenize(text, language="portuguese")
    filtered = [w for w in words if w not in stop_words]
    return " ".join(filtered)
#carregar_dados().text_pt = carregar_dados().text_pt.apply(rem_stopwords)
#print('\n\n Remover Artigos\n'+' '.join(limpaArtTxt))

#
def stem_txt(text):
    ss = SnowballStemmer('portuguese')
    words = word_tokenize(text, language="portuguese")
    return " ".join([ss.stem(w) for w in words])


#carregar_dados().text_pt = carregar_dados().text_pt.apply(stem_txt)

#print('\n\n tratar texto, reduzir palavras\n')

# ====== Treinamento (só roda se o arquivo for executado diretamente) ======
if __name__ == "__main__":
    data = carregar_dados()

    # Pré-processamento
    data["text_pt"] = data["text_pt"].apply(clean)
    data["text_pt"] = data["text_pt"].apply(is_special)
    data["text_pt"] = data["text_pt"].apply(to_lower)
    data["text_pt"] = data["text_pt"].apply(rem_stopwords)
    data["text_pt"] = data["text_pt"].apply(stem_txt)
    print(data.shape)
    print(data["text_pt"][0])
    data.head()
    data.info()

    #Maquina preditiva
    #criando uma bag of words

    # bag of words
    #x = np.array(data.iloc[:,0].values)
    y = np.array(data.sentiment.values)
    cv = CountVectorizer(max_features= 1000)
    x = cv.fit_transform(data.text_pt).toarray()


    print("X shape = ", x.shape)
    print("Y shape = ", y.shape)
    #print(y)
    #divisao de treinos, 20% pra teste e o restante pra treino
    trainx, testx, trainy, testy = train_test_split(x, y, test_size= 0.2,random_state= 9 )
    print("Train shapes: X = {}, y ={} ".format(trainx.shape, trainy.shape))
    print("Test shapes: X = {}, y ={} ".format(testx.shape, testy.shape))

    #Definindo modelos 

    gnb, mnb, bnb = GaussianNB(), MultinomialNB(alpha=1.0, fit_prior=True), BernoulliNB(alpha=1.0, fit_prior=True)

   # gnb.fit(trainx, trainy)
    mnb.fit(trainx, trainy)
    bnb.fit(trainx, trainy)

    #predicao e acuracia para escolher o melhor modelo

   # ypg = gnb.predict(testx)
    ypm = mnb.predict(testx)
    ypb = bnb.predict(testx)

    #escolher o que mais alto score tiver
    #print("Gaussian = ",accuracy_score(testy, ypg))
    print("Multinominal= ",accuracy_score(testy, ypm))
    print("Bernoulli = ",accuracy_score(testy, ypb))
    print(cv.get_feature_names_out()[:50])  # Mostra as primeiras 50 palavras


    #salva o modelo treinado
    pickle.dump(bnb, open('modelBr.pk1','wb'))
    # Salvar o CountVectorizer
    pickle.dump(cv, open('cvBr.pk1', 'wb'))


 