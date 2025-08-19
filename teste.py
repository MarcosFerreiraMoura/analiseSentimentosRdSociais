import pickle
from analiseSentimentos import clean, is_special, to_lower, rem_stopwords,stem_txt
# Carrega o vocabulário e o modelo treinado
cv = pickle.load(open('cv.pk1', 'rb'))
bnb = pickle.load(open('model0.pk1', 'rb'))


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
The show is set in a future where giant spaceships are manufactured and sent out beyond the Earth into deep space and there is no mechanism to detect when one of those ships is hurtling back towards Earth? No defence mechanism? At least in the Expanse the rock hurtling people had to figure out a way to disable the defences. Lazy and contrived.

And not sure if that's worse or better than the ship crash lands on Earth and they send children in synthetic bodies with no hazmat team? No attempt to contain or quarantine before. Just go right in there and start exploring guys no need to worry about what might be in there!!

Pahleez.

"""
y_pred = prever_sentimento(rev, cv, bnb )
print("predict teste: ", y_pred)