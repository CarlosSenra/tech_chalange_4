import requests
import logging
from typing import Optional
import pandas as pd
import os 
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

class Ingestion():
    def __init__(self,ticker:str, range_period:str, interval:str, raw_path:str, staging_path:str):
        self.ticker=ticker
        self.range_period=range_period
        self.interval=interval
        self.raw_path=raw_path
        self.staging_path=staging_path

    def save_raw_data(self) -> None:
        df = self.get_stock_data_brapi()
        df.to_csv(Path(self.raw_path, f'{self.ticker}.csv').resolve())

    def get_stock_data_brapi(self):
        """
        Função SIMPLES para coletar dados e treinar modelo de previsão
        """
        
        # Buscar dados
        url = f"https://brapi.dev/api/quote/{self.ticker}"
        logger.info(f"Buscando dados para o ticker {self.ticker}")
        params = {"range": self.range_period, "interval": self.interval}
        logger.info(f"Parâmetros: {params}")
        response = requests.get(url, params=params)
        data = response.json()
        result = data['results'][0]
        
        # Converter para DataFrame
        df = pd.DataFrame(result['historicalDataPrice'])
        df['date'] = pd.to_datetime(df['date'], unit='s')
        df = df.set_index('date')
        df = df.sort_index()
        
        # Renomear apenas as colunas que precisamos
        df = df.rename(columns={
            0: 'open',
            1: 'high', 
            2: 'low',
            3: 'close',
            4: 'volume'
        })
        
        # Manter apenas as colunas importantes
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        # Features: últimos 5 dias de preço
        for i in range(1, 6):
            df[f'close_lag_{i}'] = df['close'].shift(i)
        
        # Target: preço de amanhã
        df['target'] = df['close'].shift(-1)
        
        # Remover NaN
        #df = df.dropna()
        
        if df is not None:
            print(f"✓ {len(df)} registros coletados")
            print(f"Período: {df.index[0].date()} até {df.index[-1].date()}")
            self.date_inicial = df.index[0].date()
            self.date_final = df.index[-1].date()
            
            return df
        else:
            return None 

if __name__ == '__main__':

    getter = Ingestion("ITUB4", range_period="1y")
    getter.save_data('raw',f'ITUB4')
    