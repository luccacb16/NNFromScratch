# NN From Scratch




## Descrição
Uma biblioteca de redes neurais simples para classificação em 3 linguagens diferentes: C, C++ e Python

Em C e C++ todas as operações matriciais e processamento de dados foram implementados do 0

Em Python foi utilizada apenas a biblioteca NumPy. As funções, parâmetros e retornos foram implementados para serem similares à biblioteca sklearn (scikit-learn)

## Setup

As bibliotecas em C e C++ não possuem dependências além do compilador para as linguagens (gcc e g++)

### Python:

Clone o repositório para sua máquina
```
git clone https://github.com/luccacb16/NNFromScratch.git
```

Acesse a raiz do projeto
```
cd NNFromScratch
```

Instale as dependências (apenas numpy)
```
pip install -r requirements.txt
```
ou
```
pip install numpy
```

## Uso

Para as as linguagens C e C++, os datasets devem ser arquivos de texto, em que a primeira linha deve conter o número da quantidade de dados e na linha seguinte a quantidade de colunas (atributos) de dataset seguido de 1 (Ex: 784 1). A coluna que indica o rótulo da classe deve ser a última.

### C:
Inclua o arquivo da biblioteca:
```
#include "NN.h"
```

Crie uma variável do tipo 'NN' que receberá o retorno da função 'initNN', que possui os parâmetros:
    - taminput (int): Quantidade de colunas (atributos) do dataset
    - numhidden (int): Número de camadas escondidas desejadas
    - numoutput (int): Número de classes

```
NN Net = initNN(taminput, numhidden, numoutput);
```

Para treinar o modelo utilize a função sem retorno 'trainNN', que possui os parâmetros:
    - filename (str): Caminho para um arquivo de texto do dataset de treino
    - Net (NN): Variável que carrega a estrutura da rede neural criada no passo anterior
    - learningrate (double): Parâmetro de aprendizado (recomendado: 0.001)
```
trainNN('train.txt', Net, 0.001);
```

Para testar e obter a acurácia do modelo utilize a função 'testNN', que possui os parâmetros:
    - filename (str): Caminho para um arquivo de texto do dataset de teste
    - Net (NN): Variável que carrega a estrutura da rede neural
    - wandbfile (str): Caminho para um arquivo de texto que armazenará as matrizes de weights e biases caso a acurácia seja >= 90%
```
testNN('test.txt', Net, 'wandb.txt');
```

### C++:
Inclua o arquivo da biblioteca:
```
#include "NN.h"
```

Adicione o namespace 'neuralnetwork'
```
using namespace neuralnetwork
```

Crie um objeto NN com os parâmetros:
    - taminput (int): Quantidade de colunas (atributos) do dataset
    - numhidden (int): Número de camadas escondidas desejadas
    - numoutput (int): Número de classes
```
NN Net(taminput, numhidden, numoutput);
```

Para treinar o modelo utilize a função sem retorno 'trainNN', que possui os parâmetros:
    - filename (str): Caminho para um arquivo de texto do dataset de treino
    - func (ponteiro de função): Função de ativação desejada (ReLU, Sigmoid, anh)
    - learningrate (double): Parâmetro de aprendizado (recomendado: 0.001)
```
Net.trainNN('train.txt', ReLU, 0.001)
```

Para testar e obter a acurácia do modelo utilize a função 'testNN', que possui os parâmetros:
    - filename (str): Caminho para um arquivo de texto do dataset de teste
    - wandbfile (str): Caminho para um arquivo de texto que armazenará as matrizes de weights e biases caso a acurácia seja >= 90%
```
Net.testNN('test.txt', 'wandb.txt')
```

### Python:
Importe a biblioteca
```
from nn import *
```

Carregue o arquivo csv do dataset num pandas DataFrame e separe entre dados de input (atributos) e o rótulo. Utilize a função 'split' para dividí-los em treino e teste. Ela possui como parâmetros:
    - x (np.ndarray): Atributos
    - y (np.ndarray): Rótulos
    - test_size (float): Porcentagem do dataset que será de teste (default = 0.2)[
A função split retorna 4 np.ndarrays: x_train, y_train, x_test, y_test, que são os pares de atributos-rótulos de treino e teste, respectivamente
```
data = pd.read_csv('scripts/mnist.csv')

y = data['label']
x = data.drop('label', axis=1)

x_train, y_train, x_test, y_test = split(x, y, test_size=0.2) 
```

Crie um objeto NN com os parâmetros:
    - numhidden (int): Quantidade de camadas escondidas desejadas (default = 10)
    - activation (str): Função de ativação (ReLU, Sigmoid, tanh) (default = ReLU)
    - epochs (int): Número de iterações para o treinamento (default = 100)
```
nn = NN(numhidden=10, activation='relu', epochs=100)
```

Utilize a função 'train' para treinar o modelo. Parâmetros:
    - X (np.ndarray): X de treinamento (x_train)
    - Y (np.ndarray): Y de treinamento (y_train)
    - lr (float): Learning rate (default = 0.001)
A função train retorna o objeto do modelo treinado

Utilize a função 'predict' para obter as predições dos rótulos de cada caso de teste. Parâmetros:
    - X (np.ndarray): Atributos de teste (x_test)
A função predict retorna uma lista com as predições (número da classe)

Para obter a acurácia do modelo, utilize a função 'score', com parâmetros:
    - predictions (list): Predições obtidas no retorno da função 'predict'
    - Y (np.ndarray): Rótulos de teste (y_test)
A função score retorna um float que é a porcentagem de acurácia (0 <= acurácia <= 1)
```
model = nn.train(x_train, y_train, lr=0.001)
predictions = model.predict(x_test)

accuracy = model.score(predictions, y_test)
```

Para um teste rápido, basta executar o arquivo 'main.py' da pasta 'scripts'