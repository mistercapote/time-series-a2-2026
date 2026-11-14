from sklearn.base import BaseEstimator, RegressorMixin
import torch.nn as nn

class MLPTorch(BaseEstimator, RegressorMixin):
    def __init__(self,
                 neuronios=64,
                 camadas=2,
                 lr=1e-3,
                 weight_decay=0.0,
                 epocas=200,
                 paciencia=20,
                 n_es=28,
                 seed=0):
        
        self.neuronios = neuronios
        self.camadas = camadas
        self.lr = lr
        self.weight_decay = weight_decay
        self.epocas = epocas
        self.paciencia = paciencia
        self.n_es = n_es
        self.seed = seed
        
def _criar_rede(n_entradas, neuronios, camadas):
    pecas = []
    tamanho = n_entradas
    for _ in range(camadas):
        # TODO: explicar por que escolhemos ReLU, e não testamos outras ativações
        pecas += [nn.Linear(tamanho, neuronios), nn.ReLU()]
        tamanho = neuronios
        
    pecas.append(nn.Linear(tamanho, 1)) # Saída de um único valor desejado
    return nn.Sequential(*pecas)