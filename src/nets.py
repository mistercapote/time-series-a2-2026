from sklearn.base import BaseEstimator, RegressorMixin
import torch.nn as nn
import numpy as np
import torch

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

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
    
        # padronização: média e desvio só dos dados recebidos
        self.mu_x_ = X.mean(axis=0)
        self.sd_x_ = X.std(axis=0)
        self.sd_x_[self.sd_x_ == 0] = 1.0  # coluna constante não divide por zero
        self.mu_y_ = y.mean()
        self.sd_y_ = y.std() or 1.0
        
        # Dados padronizados
        Xp = (X - self.mu_x_) / self.sd_x_
        yp = (y - self.mu_y_) / self.sd_y_
        
        # últimos n_es dias = fatia de early stopping (sem embaralhar)
        X_tr, y_tr = Xp[:-self.n_es], yp[:-self.n_es]
        X_es, y_es = Xp[-self.n_es:], yp[-self.n_es:]
        print(f"treino: {len(X_tr)} linhas | early stopping: {len(X_es)} linhas")
        return self
    
    