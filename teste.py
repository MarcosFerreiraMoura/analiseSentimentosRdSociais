import pickle
from analiseSentimentos import clean, is_special, to_lower, rem_stopwords,stem_txt
# Carrega o vocabulário e o modelo treinado
cv = pickle.load(open('cvBr.pk1', 'rb'))
bnb = pickle.load(open('modelBr.pk1', 'rb'))


def prever_sentimento(texto, cv, modelo):
    #pre- processamento
    f1 = clean(texto)
    f2 = is_special(f1)
    f3 = to_lower(f2)
    print(f3)
    f4 = rem_stopwords(f3)
    f5 = stem_txt(f4)

    vetor =  cv.transform([f5]).toarray()
    return modelo.predict(vetor)

#fazendo o teste

rev = """
filme ruim, eu estou muito triste em ver que meu filme favorito da minha infancia se tornou algo tão banal, complicado! 

"""
y_pred = prever_sentimento(rev, cv, bnb )
print("predict teste: ", y_pred)
print(cv.get_feature_names_out()[:50])  # Mostra as primeiras 50 palavras
