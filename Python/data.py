import numpy as np

def split(x: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> tuple:
  '''
  Divide o conjunto de dados em treino e teste de acordo com a proporção definida
  
  Args:
    x: Conjunto de dados
    y: Conjunto de labels
    test_size: Proporção de dados para teste
    
  Returns:
    x_train: Conjunto de dados de treino
    y_train: Conjunto de labels de treino
    x_test: Conjunto de dados de teste
    y_test: Conjunto de labels de teste
  '''
  
  x = np.array(x)
  y = np.array(y)
    
  # Embaralha da mesma forma
  p = np.random.permutation(x.shape[0])
  x = x[p]
  y = y[p]
    
  index = int(x.shape[0] * (1-test_size))
    
  x_train = x[:index]
  y_train = y[:index]
    
  x_test = x[index:]
  y_test = y[index:]
    
  return x_train, y_train, x_test, y_test