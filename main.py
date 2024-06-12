import matplotlib.pyplot as plt
import pandas as pd
import sklearn.neural_network as skl
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

qdt_items_teste = 186


def preparar_dados(data):
    atributos = []
    classe = []

    # Os primeiros valores foram reservados para testes com base no valor da variavel qdt_items_treino
    for i in range(qdt_items_teste, data.shape[0]):
        atributos.append([data.iloc[i]['Defeito'], data.iloc[i]['Rustificacao'], data.iloc[i]['Acicula']])
        classe.append(data.iloc[i]['Classe'])

    return atributos, classe


def treinar_rede_neural(atributos, classes):
    # DOC: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    classificador = skl.MLPClassifier()
    classificador.fit(atributos, classes)
    return classificador


def testar_modelo(classificador, data):
    acertos = 0
    previsao = []
    valor_verdadeiro = []

    for i in range(qdt_items_teste):
        defeito = data.iloc[i]['Defeito']
        rustificacao = data.iloc[i]['Rustificacao']
        acicula = data.iloc[i]['Acicula']
        classe = data.iloc[i]['Classe']

        classe_prevista = classificador.predict([[defeito, rustificacao, acicula]])
        previsao.append(classe_prevista[0])
        valor_verdadeiro.append(classe)

        if classe_prevista[0] == classe:
            acertos += 1

    precision, recall, fscore, _ = precision_recall_fscore_support(valor_verdadeiro, previsao, average='micro')

    print(f'\n-------------------------------------------')
    print(f'\nQuantidade de iems de teste: {qdt_items_teste}')
    print(f'\nQuantidade de items de treino: {data.shape[0] - qdt_items_teste}')
    print(f'\nPorcentagem de acertos: {(acertos / qdt_items_teste) * 100} %')
    print(f'\n-------------------------------------------')
    print(f'\nPrecisão : {precision}')
    print(f'\nRecall : {recall}')
    print(f'\nFscore : {fscore}')
    print(f'\n-------------------------------------------')
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(valor_verdadeiro, previsao)).plot()
    plt.show()


if __name__ == '__main__':
    data = pd.read_excel('data/mudas_pinus.xlsx')
    atributos, classes = preparar_dados(data)
    classificador = treinar_rede_neural(atributos, classes)
    testar_modelo(classificador, data)
