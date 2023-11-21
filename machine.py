import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def carregar():
    # Carregar o conjunto de dados
    data = load_breast_cancer()
    X = data.data
    y = data.target
    return X, y
# Dividir o conjunto de dados em treino e teste
X, y = carregar()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def treinar(classificador, parametros, X_train, y_train, X_test, y_test):
    if classificador == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=parametros['n_neighbors'])
    elif classificador == 'SVM':
        clf = SVC(kernel=parametros['kernel'], degree=parametros['degree'])
    elif classificador == 'DT':
        clf = DecisionTreeClassifier(max_depth=parametros['max_depth'])
    elif classificador == 'RF':
        clf = RandomForestClassifier(n_estimators=parametros['n_estimators'], max_depth=parametros['max_depth'])
    else:
        raise ValueError("Classificador não encontrado.")
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    matriz_confusao = confusion_matrix(y_test, y_pred)
    
    # Salvando a matriz de confusão como imagem
    nome_imagem = f'confusion_matrix_{classificador}.png'
    caminho_imagem = f'static/{nome_imagem}'
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax)
    plt.savefig(caminho_imagem)
    plt.close(fig)

    return nome_imagem