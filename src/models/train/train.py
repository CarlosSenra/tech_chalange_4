import logging
import mlflow
from src.models.LSTM.SimpleLSTM import SimpleLSTM
from src.data.feature_eng import FeatureEng

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

class Train():
    def __init__(self, model:SimpleLSTM,
                    feature_eng:FeatureEng, 
                    epochs:int, 
                    batch_size:int, 
                    validation_split:float, 
                    verbose:int, 
                    metric_list:list[str], 
                    optimizer:str, 
                    loss:str,
                    experiment_name:str,
                    ):
        self.model = model
        self.feature_eng = feature_eng
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose
        self.metric_list = metric_list
        self.optimizer = optimizer
        self.loss = loss
        self.experiment_name = experiment_name
    def run(self):
        try:
            self.__apply_feature_eng()
            self.__mlflow_train()
        except Exception as e:
            logger.error(f"Erro ao executar o treinamento: {e}")
            raise e

    def __apply_feature_eng(self):
        self.X_train, self.y_train, self.X_test, self.y_test = self.feature_eng.run(janela=5)

    def __mlflow_train(self):
        mlflow.tensorflow.autolog()
        mlflow.set_experiment(self.experiment_name)
        try:
            with mlflow.start_run():
                self.model.compile(optimizer=self.optimizer, loss=self.loss)
                self.model.fit(self.X_train, self.y_train, 
                               epochs=self.epochs, 
                               batch_size=self.batch_size, 
                               validation_split=self.validation_split,
                               verbose=self.verbose)
                mlflow.tensorflow.log_model(self.model, "model")
                mlflow.evaluate(self.model, self.X_test, self.y_test, metric_list=self.metric_list)
                mlflow.end_run()
        except Exception as e:
            logger.error(f"Erro ao treinar o modelo: {e}")
            raise e