import cv2, os, re
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
'''0 import classificadores'''
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


diretorio= "./database/"

"""1 funçao natural_sort"""
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

'''2 lista pastas e ordena'''
content = os.listdir(diretorio)
content = natural_sort(content)

'''3 vetores treino e teste'''
treinoImagem = []
treinoRotulo = []
testeImagem = []
testeRotulo = []

contpessoa = 1

'''4 Para cada pessoa nos diretorios, imprima a pasta delas'''
for pessoa in content:
    diretorioPessoas = diretorio + pessoa + '/'
    if not os.path.isdir(diretorioPessoas): continue

    print("Operando pessoa", contpessoa)
    contpessoa += 1
    # print("Operando pessoa", diretorioPessoas)

    contentPessoas = os.listdir(diretorioPessoas)
    contentPessoas = natural_sort(contentPessoas)

    '''6 contador pra condição da metade para o treino e metade pro teste'''
    contador = 1;

    '''7 Para o contador (indicador) na pasta das pessoas, faça:'''
    for img in contentPessoas:
        nomeImg = diretorioPessoas + img
        #converte a imagem para cinza
        imagemCinza = cv2.cvtColor(cv2.imread(nomeImg), cv2.COLOR_RGB2GRAY)

        pontos = 8
        raio = 2
        metodo = "nri_uniform"

        lbp = local_binary_pattern(imagemCinza, pontos, raio, method = metodo )
        # print(lbp)

        x = itemfreq(lbp)
        somaTotal = sum(x[:, 1])

        '''8 se o contador (lá em cima no passo 6) for menor que dois, adicione 2
        imagens para o treinamento img (passo 3)'''
        if contador <=2: #quantidade de amostras no treino
            treinoImagem.append(x[:, 1]/somaTotal)
            treinoRotulo.append(pessoa)
        else:
            #o restante é do teste
            testeImagem.append(x[:, 1]/somaTotal)
            testeRotulo.append(pessoa)

        contador = contador + 1

'''objetos dos classificadores'''


classKNN = KNeighborsClassifier(n_neighbors = 1)
#manda o classificador aprender
'''treinamento'''
print("Resultados do KNN")
classKNN.fit(treinoImagem, treinoRotulo)
'''classificacao'''
predicao = classKNN.predict(testeImagem)
#print(predicao)

acuracia = classKNN.score(testeImagem, testeRotulo)
print(acuracia*100)

'''RANDOM FOREST'''
print("Resultados do Random Forest")
from sklearn.ensemble import RandomForestClassifier
classRN = RandomForestClassifier()
# classRN = RandomForestClassifier(n_estimators=40, max_depth=2, random_state=0)
# utilizando o RN com os parametros padrao, deu *** porcentagem
classRN.fit(treinoImagem, treinoRotulo)
predicao = classRN.predict(testeImagem)
print(predicao)

acuracia = classRN.score(testeImagem, testeRotulo)
print(acuracia*100)

#não precisa renomear as imagens, só jogar dentro da pasta pessoas
