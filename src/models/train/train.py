import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.sklearn
from src.models.LSTM.SimpleLSTM import SimpleLSTM
from src.data.feature_eng import FeatureEng

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

def mape_metric(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    epsilon = tf.keras.backend.epsilon()
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def rmse_metric(y_true, y_pred):
    """Root Mean Squared Error"""
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


class Train():
    def __init__(self, model:SimpleLSTM,
                    feature_eng:FeatureEng, 
                    epochs:int, 
                    batch_size:int, 
                    validation_split:float, 
                    verbose:int, 
                    metric_list:list[str], 
                    validation_metrics:list[str],
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
        self.validation_metrics = validation_metrics
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
                    "model_list_units": str(self.model.list_units),
                    "model_output_dim": self.model.output_dim,
                    "model_num_layers": len(self.model.list_units),
                })
                metrics_to_compile = list(self.metric_list)
                
                if "mape" in self.validation_metrics:
                    metrics_to_compile.append(mape_metric)
                if "rmse" in self.validation_metrics:
                    metrics_to_compile.append(rmse_metric)
                
                self.model.compile(
                    optimizer=self.optimizer,
                    loss=self.loss,
                    metrics=metrics_to_compile,
                )
                
                history = self.model.fit(
                    self.X_train, self.y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_split=self.validation_split,
                    verbose=self.verbose,
                )
                
                for epoch in range(self.epochs):
                    metrics_to_log = {
                        "loss": history.history["loss"][epoch],
                        "val_loss": history.history["val_loss"][epoch],
                    }
                    
                    for metric in self.validation_metrics:
                        possible_keys = [
                            f"val_{metric}",
                            f"val_{metric}_metric",
                            f"val_{metric}_1",
                        ]
                        for key in possible_keys:
                            if key in history.history:
                                metrics_to_log[f"val_{metric}"] = history.history[key][epoch]
                                break
                    
                    mlflow.log_metrics(metrics_to_log, step=epoch)
                
                mlflow.sklearn.log_model(
                    self.feature_eng.scaler_X, "scaler_X"
                )
                mlflow.sklearn.log_model(
                    self.feature_eng.scaler_y, "scaler_y"
                )
                
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