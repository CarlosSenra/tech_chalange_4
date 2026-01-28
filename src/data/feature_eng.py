import logging
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

class FeatureEng():
    def __init__(self, raw_path:str, staging_path:str):
        self.raw_path=raw_path
        self.staging_path=staging_path
        self.scaler = StandardScaler()
        self.features = ['open', 'high', 'low', 'close', 'volume']

    def create_features(self):
        df = pd.read_csv(Path(self.raw_path, f'{self.ticker}-{self.date_inicial}---{self.date_final}.csv'))
        
        df = df.fillna(0)
        self.data = df[self.features].values




    def criar_sequencias(self,dados:np.array, janela:int=5) -> tuple[np.array, np.array]:
        dados = self.scaler.fit_transform(dados)
        X, y = [], []
        for i in range(len(dados) - janela):
            X.append(dados[i:i+1])
            y.append(dados[i:i+janela,3]) # target é o preço de fechamento
        self.X = np.array(X)
        self.y = np.array(y)
        
        return self.X, self.y