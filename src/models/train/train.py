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
        mlflow.set_experiment(self.experiment_name)
        try:
            with mlflow.start_run():
                mlflow.log_params({
                    "optimizer": self.optimizer,
                    "loss": self.loss,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "validation_split": self.validation_split,
                })
                self.model.compile(
                    optimizer=self.optimizer,
                    loss=self.loss,
                    metrics=self.metric_list,
                )
                history = self.model.fit(
                    self.X_train, self.y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_split=self.validation_split,
                    verbose=self.verbose,
                )
                for epoch, (loss_val, val_loss_val) in enumerate(zip(
                    history.history["loss"],
                    history.history["val_loss"],
                )):
                    mlflow.log_metrics({"loss": loss_val, "val_loss": val_loss_val}, step=epoch)
                mlflow.tensorflow.log_model(self.model, "model")

                eval_results = self.model.evaluate(
                    self.X_test, self.y_test, verbose=0
                )
                eval_metrics = dict(
                    zip(self.model.metrics_names, eval_results)
                )
                mlflow.log_metrics(
                    {f"test_{k}": v for k, v in eval_metrics.items()}
                )
        except Exception as e:
            logger.error(f"Erro ao treinar o modelo: {e}")
            raise e