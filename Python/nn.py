import numpy as np
from data import *
from functions import *

# NN
class NN:
    '''
    Classe da rede neural
    
    Args:
        numhidden (int): Número de neurônios na camada escondida
        activation (function): Função de ativação
        epochs (int): Número de iterações do treinamento
    '''
    
    def __init__(self, numhidden: int = 10, activation: str = 'relu', epochs: int = 100):
        self.numinput = 0
        self.numhidden = numhidden
        self.numoutput = 0
        
        self.epochs = epochs
        
        self.f = activation
        self.df = 'd' + activation   
        
        self.Wi = None
        self.Bi = None
        self.Wh = None
        self.Bh = None
        
    def init(self):
        '''
        Inicializa os pesos e bias da rede neural de acordo com a inicialização de Kaiming/He
        '''
        
        self.Wi = np.random.normal(size = (self.numhidden, self.numinput)) * np.sqrt(2./(self.numinput + self.numhidden))
        self.Bi = np.zeros((self.numhidden, 1))
    
        self.Wh = np.random.normal(size = (self.numoutput, self.numhidden)) * np.sqrt(2./(self.numhidden + self.numoutput))
        self.Bh = np.zeros((self.numoutput, 1))
        
        return self.Wi, self.Bi, self.Wh, self.Bh

    def forwardprop(self, X: np.ndarray) -> tuple:
        X = X[:, np.newaxis]
        
        H = self.Wi.dot(X) + self.Bi
        fH = activation(H, self.f)

        O = self.Wh.dot(fH) + self.Bh
        sO = Softmax(O)
        
        return H, fH, O, sO

    def backprop(self, X: np.ndarray, Y: np.ndarray, H: np.ndarray, fH: np.ndarray, O: np.ndarray, sO: np.ndarray) -> tuple:
        X = X[:, np.newaxis]
        
        onehot = OneHot(Y, self.numoutput)
        
        # Wh, Bh
        Erro1 = (sO - onehot) * 2
        Gradiente1 = activation(O, self.df) * (Erro1)
        
        dWh = Gradiente1.dot(fH.T)
        dBh = Gradiente1 * 1

        # Wi, Bi
        Erro2 = self.Wh.T.dot(Erro1)
        Gradiente2 = activation(H, self.df) * (Erro2)
        
        dWi = Gradiente2.dot(X.T)
        dBi = Gradiente2 * 1

        return dWi, dBi, dWh, dBh

    def update(self, dWi: np.ndarray, dBi: np.ndarray, dWh: np.ndarray, dBh: np.ndarray, lr: float) -> tuple:
        self.Wi = self.Wi - (dWi * lr)
        self.Bi = self.Bi - (dBi * lr)
        self.Wh = self.Wh - (dWh * lr)
        self.Bh = self.Bh - (dBh * lr)
    
    def train(self, X: np.ndarray, Y: np.ndarray, lr: float = 0.001):
        '''
        Treina a rede neural com os dados de entrada X e saída Y
        '''
        
        n = X.shape[0]
        
        self.numinput = X[0].shape[0]
        self.numoutput = len(np.unique(Y))

        # Inicializa os parâmetros
        self.Wi, self.Bi, self.Wh, self.Bh = self.init() 
        
        # Roda 'epochs' treinamentos para 'n' dados
        for _ in range(self.epochs):
  
            # Embaralha a cada epoch
            p = np.random.permutation(X.shape[0])
            X = X[p]
            Y = Y[p]
            
            for i in range(n):
                H, fH, O, sO = self.forwardprop(X[i])
                dWi, dBi, dWh, dBh = self.backprop(X[i], Y[i], H, fH, O, sO)
                self.update(dWi, dBi, dWh, dBh, lr)
        
        # Retorna o proprio objeto
        return self
        
    def predict(self, X: np.ndarray) -> list:
        '''
        Retorna uma lista com as predições para cada exemplo de X
        '''
        
        n = X.shape[0]
        
        predictions = []
        
        for i in range(n):
            _, _, _, sO = self.forwardprop(X[i])
            predictions.append(np.argmax(sO))
            
        return predictions
    
    def accuracy(self, predictions: list, Y: np.ndarray) -> float:
        '''
        Retorna a acurácia do modelo (número de acertos / número de exemplos)
        '''
        
        if len(predictions) != len(Y):
            raise ValueError('predictions and Y must have the same length')
        
        return np.sum(predictions == Y) / len(Y)
    
    def precision(self, predictions: list, Y: np.ndarray) -> float:
        '''
        Retorna a precisão do modelo (número de verdadeiros positivos / número de positivos)
        '''
        if len(predictions) != len(Y):
            raise ValueError('predictions and Y must have the same length')
        
        classes = len(np.unique(Y))
        precisions = []
        
        for i in range(classes):
            TP = np.sum((predictions == np.full_like(predictions, i)) & (Y == i))
            FP = np.sum((predictions == np.full_like(predictions, i)) & (Y != i))
            
            precisions.append(TP / (TP + FP))
        
        return np.mean(precisions)
        