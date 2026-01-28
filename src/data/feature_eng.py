from ast import Str
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
    def __init__(self, raw_path:str, ticker:str, features_X:list[str], features_y:list[str]):
        self.raw_path=raw_path
        self.ticker=ticker
        self.features_X = features_X
        self.features_y = features_y
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        logger.debug(f"Iniciando Feature Engineering para o ticker {self.ticker}")
        self.__reception_data()

    def __reception_data(self):
        try:
            logger.debug(f"raw_path:{self.raw_path}")
            logger.debug(f"ticker:{self.ticker}")
            logger.debug(f"features_X:{self.features_X[0]}")
            logger.debug(f"features_y:{self.features_y[0]}")
            df = pd.read_csv(Path(self.raw_path, f'{self.ticker}.csv'))
            df = df.fillna(0)
            self.data_X = df[self.features_X].values
            self.data_y = df[self.features_y].values
            logger.debug(f"data_X shape:{self.data_X.shape}")
            logger.debug(f"data_y shape:{self.data_y.shape}")
        except Exception as e:
            logger.error(f"Erro ao receber dados para o ticker {self.ticker}: {e}")
            raise e

    def create_sequences(self, janela:int) -> tuple[np.array, np.array]:
        try:
            self.scaled_data_X = self.scaler_X.fit_transform(self.data_X)
            self.scaled_data_y = self.scaler_y.fit_transform(self.data_y)

            X, y = [], []
            for i in range(len(self.scaled_data_X) - janela):
                X.append(self.scaled_data_X[i:i+1])
                y.append(self.scaled_data_y[i:i+janela]) # target é o preço de fechamento
            self.X = np.array(X)
            self.y = np.array(y)
            logger.debug(f"X shape:{self.X.shape}")
            logger.debug(f"y shape:{self.y.shape}")
        except Exception as e:
            logger.error(f"Erro ao criar sequências para o ticker {self.ticker}: {e}")
            raise e
        return self.X, self.y

    def reverse_sequences(self, _X:np.array, _y:np.array) -> tuple[np.array, np.array]:
        try:
            logger.debug(f"_X shape:{_X.shape}")
            logger.debug(f"_y shape:{_y.shape}")
            __X = self.scaler_X.inverse_transform(_X.reshape(-1, _X.shape[-1]))
            __y = self.scaler_y.inverse_transform(_y.reshape(1, _y.shape[-1]))
            logger.debug(f"__X shape:{__X.shape}")
            logger.debug(f"__y shape:{__y.shape}")
        except Exception as e:
            logger.error(f"Erro ao inverter sequências para o ticker {self.ticker}: {e}")
            raise e
        return __X, __y

