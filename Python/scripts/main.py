import pandas as pd

# Importa as bibliotecas necessárias
from data import *
from functions import *
from nn import *

# Carrega os dados para um pandas dataframe
data = pd.read_csv('mnist.csv')

# Separa os dados de input e output
y = data['label']
x = data.drop('label', axis=1)

# Separa os dados em treino e teste
x_train, y_train, x_test, y_test = split(x, y, test_size=0.2)

# Normaliza os dados de treino e teste
x_train = normalize(x_train, 'mnist')
x_test = normalize(x_test, 'mnist')

nn = NN(numhidden=20, activation='relu', epochs=100) # Cria o modelo
model = nn.train(x_train, y_train, lr=0.001) # Treina
predictions = model.predict(x_test) # Predições

accuracy = model.accuracy(predictions, y_test) # Calcula a acurácia
precision = model.precision(predictions, y_test) # Calcula a precisão

print(accuracy)
print(precision)