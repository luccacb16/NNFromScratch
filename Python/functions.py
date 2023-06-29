import numpy as np

# Normalizações 
def minmax(X):
  return (X - np.max(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0) + 1e-7)

def zscore(X):
  return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def l1(X):
  return X / np.linalg.norm(X, ord=1, axis=0)

def mnist(X):
  return X / 255.

def greyscale(X):
  return X / 255.

def normalize(X, f: str):
  norms = ['minmax', 'zscore', 'l1', 'mnist', 'greyscale']
  
  if f.lower() in norms:
    return eval(f.lower())(X)
  else:
    raise ValueError(f'Invalid normalization function: {f}')

# Funções de ativação
def activation(X, f: str):
  act = ['relu', 'sigmoid', 'tanh', 'drelu', 'dsigmoid', 'dtanh']
  
  if f.lower() in act:
    return eval(f.lower())(X)
  else:
    raise ValueError(f'Invalid activation function: {f}')

def relu(X):
  return np.maximum(X, 0)

def drelu(X):
  return X > 0

def sigmoid(X):
  return 1 / (1 + np.exp(-X))

def dsigmoid(X):
  return 1 - sigmoid(X)

def tanh(X):
  return np.tanh(X)

def dtanh(X):
  return 1 - np.tanh(X)**2

def OneHot(label, numoutput):
  label = int(label)
    
  onehot = np.zeros((numoutput, 1))
  onehot[label] = 1
    
  return onehot

def Softmax(X):
  exp = np.exp(X - np.max(X)) 
        
  return exp / exp.sum(axis=0)