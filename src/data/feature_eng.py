from ast import Str
import logging
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

class FeatureEng():
    def __init__(self, raw_path:str, ticker:str, features_X:list[str], features_y:list[str], test_size:float):
        self.raw_path=raw_path
        self.ticker=ticker
        self.features_X = features_X
        self.features_y = features_y
        self.test_size = test_size
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        logger.debug(f"Iniciando Feature Engineering para o ticker {self.ticker}")
        self.__reception_data()


    def run(self, janela:int) -> tuple[np.array, np.array, np.array, np.array]:
        try:
            logger.info(f"Iniciando Feature Engineering para o ticker {self.ticker}")
            self.__reception_data()
            self.__split_data()
            self.X_train, self.y_train = self.__create_sequences(janela, self.scaled_X_train, self.scaled_y_train)
            self.X_test, self.y_test = self.__create_sequences(janela, self.scaled_X_test, self.scaled_y_test)
            
            return self.X_train, self.y_train, self.X_test, self.y_test
        except Exception as e:
            logger.error(f"Erro ao executar Feature Engineering para o ticker {self.ticker}: {e}")
            raise e

    def reverse_sequences(self, _X:np.array, _y:np.array) -> tuple[np.array, np.array]:
        logger.debug(f"_X shape:{_X.shape}")
        logger.debug(f"_y shape:{_y.shape}")
        try:
            __X = _X.reshape(-1, _X.shape[-1])
            __X = self.scaler_X.inverse_transform(__X)
            
            __y = _y.reshape(-1, _y.shape[-2])
            __y = self.scaler_y.inverse_transform(__y)
            
            logger.debug(f"__X shape:{__X.shape}")
            logger.debug(f"__y shape:{__y.shape}")
            return __X, __y
        except Exception as e:
            logger.error(f"Erro ao inverter sequências para o ticker {self.ticker}: {e}")
            raise e

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


    def __split_data(self) -> tuple[np.array, np.array, np.array, np.array]:
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_X, self.data_y, test_size=self.test_size, random_state=42)
            
            logger.debug(f"X_train shape:{self.X_train.shape}")
            logger.debug(f"X_test shape:{self.X_test.shape}")
            logger.debug(f"y_train shape:{self.y_train.shape}")
            logger.debug(f"y_test shape:{self.y_test.shape}")

            self.scaled_X_train = self.scaler_X.fit_transform(self.X_train)
            self.scaled_X_test = self.scaler_X.transform(self.X_test)

            self.scaled_y_train = self.scaler_y.fit_transform(self.y_train)
            self.scaled_y_test = self.scaler_y.transform(self.y_test)

            return self.scaled_X_train, self.scaled_X_test, self.scaled_y_train, self.scaled_y_test

        except Exception as e:
            logger.error(f"Erro ao dividir dados para o ticker {self.ticker}: {e}")
            raise e
        

    def __create_sequences(self, janela:int, scaled_data_X:np.array, scaled_data_y:np.array) -> tuple[np.array, np.array]:
        try:
            X, y = [], []
            for i in range(len(scaled_data_X) - janela):
                X.append(scaled_data_X[i:i+1])
                y.append(scaled_data_y[i:i+janela]) # target é o preço de fechamento
            X = np.array(X)
            y = np.array(y)
            logger.debug(f"X shape:{X.shape}")
            logger.debug(f"y shape:{y.shape}")
        except Exception as e:
            logger.error(f"Erro ao criar sequências para o ticker {self.ticker}: {e}")
            raise e

        return X, y