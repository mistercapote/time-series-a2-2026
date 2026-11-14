from sklearn.base import BaseEstimator, RegressorMixin
import torch.nn as nn
import numpy as np
import torch
import copy

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
    
        self.rede_ = self._criar_rede(X.shape[1], self.neuronios, self.camadas)
        self.epoca_escolhida_, self.historico_ = self._treinar(
            self.rede_, 
            X_tr, y_tr, X_es, y_es, 
            self.lr, self.weight_decay, self.epocas, self.paciencia
        )
        print(f"época escolhida: {self.epoca_escolhida_} | fatia ES: {len(X_es)} dias")
        return self
    
    def predict(self, X):
        
        # Padronizando a amostra
        X = np.asarray(X, dtype=np.float32)
        Xp = (X - self.mu_x_) / self.sd_x_
        
        self.rede_.eval()
        with torch.no_grad():
            yp = self.rede_(torch.as_tensor(Xp)).numpy().ravel()
         
        # Retorna despadronizado   
        return yp * self.sd_y_ + self.mu_y_  
        
    @staticmethod    
    def _criar_rede(n_entradas, neuronios, camadas):
            pecas = []
            tamanho = n_entradas
            for _ in range(camadas):
                # TODO: explicar por que escolhemos ReLU, e não testamos outras ativações
                pecas += [nn.Linear(tamanho, neuronios), nn.ReLU()]
                tamanho = neuronios
                
            pecas.append(nn.Linear(tamanho, 1)) # Saída de um único valor desejado
            return nn.Sequential(*pecas)        

    @staticmethod      
    def _treinar(
                rede,
                X_tr, y_tr,
                X_es, y_es,
                lr,
                weight_decay,
                epocas,
                paciencia,
                batch=64
                ):
        
        X_tr, X_es = torch.as_tensor(X_tr), torch.as_tensor(X_es)
        y_tr = torch.as_tensor(y_tr).reshape(-1, 1)
        y_es = torch.as_tensor(y_es).reshape(-1, 1)
        
        # Método de otimização e função de perdão padrões
        opt = torch.optim.Adam(
            rede.parameters(), lr=lr, weight_decay=weight_decay
        )
        mse = nn.MSELoss()
        
        melhor, sem_melhora, epoca_melhor = float("inf"), 0, 0
        melhor_estado = copy.deepcopy(rede.state_dict())
        historico = []
        
        for epoca in range(1, epocas + 1):
            # Treino com mini-batches embaralhados dentro do conj. de treino
            rede.train()
            tr_size = len(X_tr)
            ordem = torch.randperm(tr_size)
            soma = 0.0
            
            for i in range(0,tr_size, batch):
                
                # Obtendo o índice do batch de dados
                idx = ordem[i:i + batch]
                
                # Treino
                loss = mse(rede(X_tr[idx]), y_tr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                soma += loss.item() * len(idx)
                
            perda_tr = soma / tr_size
        
            rede.eval()
            with torch.no_grad():
                # Perda no conjunto de early_stopping
                perda_es = mse(rede(X_es), y_es).item()
                
            historico.append((perda_tr, perda_es))
            
            # Ciclo de early stopping  
            if perda_es < melhor:
                melhor, sem_melhora, epoca_melhor = perda_es, 0, epoca
                melhor_estado = copy.deepcopy(rede.state_dict())
            else:
                sem_melhora += 1
                if sem_melhora >= paciencia:
                    break  
        
        # Obtém o melhor estado e retorna ele        
        rede.load_state_dict(melhor_estado)
        return epoca_melhor, historico
            
        