import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


class Evaluate():
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.client = mlflow.tracking.MlflowClient()
        self.experiment = self.client.get_experiment_by_name(experiment_name)
        if self.experiment is None:
            raise ValueError(f"Experimento '{experiment_name}' não encontrado")

    def get_best_model(self, metric: str = "val_mae") -> str:
        runs = self.client.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{metric} ASC"],
            max_results=1,
        )
        
        if not runs:
            raise ValueError(f"Nenhum run encontrado no experimento '{self.experiment_name}'")
        
        best_run = runs[0]
        best_run_id = best_run.info.run_id
        
        metric_value = best_run.data.metrics.get(metric, None)
        logger.info(
            f"Melhor modelo encontrado: run_id={best_run_id}, "
            f"{metric}={metric_value}"
        )
        
        return best_run_id

    def get_model_name_by_run_id(self, run_id: str) -> str:
        """
        Busca o nome do modelo registrado usando o run_id.
        
        Args:
            run_id: ID do run do MLflow
        
        Returns:
            Nome do modelo registrado ou None se não encontrado
        """
        all_models = self.client.search_registered_models()
        
        for model in all_models:
            versions = self.client.search_model_versions(f"name='{model.name}'")
            for version in versions:
                if version.run_id == run_id:
                    logger.info(
                        f"Modelo encontrado: '{model.name}' (versão {version.version}) "
                        f"para run_id={run_id}"
                    )
                    return model.name
        
        logger.warning(f"Nenhum modelo registrado encontrado para run_id={run_id}")
        return None

    def register_and_stage(
        self,
        metric: str = "val_mae",
        model_name: str = None,
    ) -> tuple[str, int, str]:
        """
        Passo 1: Registra o melhor modelo no Model Registry (mlflow.register_model).
        Passo 2: Move a versão para Staging (client.transition_model_version_stage).
        
        Args:
            metric: Métrica de validação para selecionar o melhor modelo
            model_name: Nome do modelo registrado. Se None, usa o nome do experimento.
        
        Returns:
            Tupla (run_id, version, model_name) do modelo registrado e em Staging
        """
        best_run_id = self.get_best_model(metric)
        
        if model_name is None:
            model_name = self.get_model_name_by_run_id(best_run_id)
            if model_name is None:
                model_name = self.experiment_name
                logger.info(
                    f"Nenhum modelo registrado encontrado para run_id={best_run_id}. "
                    f"Usando nome do experimento: '{model_name}'"
                )
        
        model_uri = f"runs:/{best_run_id}/model"
        
        try:
            self.client.get_registered_model(model_name)
        except mlflow.exceptions.MlflowException:
            self.client.create_registered_model(model_name)
            logger.info(f"Modelo registrado '{model_name}' criado")
        
        result = mlflow.register_model(model_uri=model_uri, name=model_name)
        version = result.version
        logger.info(
            f"Modelo registrado: nome='{model_name}', versão={version}"
        )
        
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
        )
        logger.info(
            f"Modelo '{model_name}' versão {version} movido para Staging"
        )
        
        return best_run_id, version, model_name

    def promote_to_production(
        self,
        metric: str = "val_mae",
        model_name: str = None,
        approve_promotion: bool = True,
    ) -> str:
        """
        Registra o melhor modelo, move para Staging e, se aprovado,
        arquiva versões em Production e promove a nova para Production.
        
        Args:
            metric: Métrica de validação para selecionar o melhor modelo
            model_name: Nome do modelo registrado. Se None, usa o nome do experimento.
            approve_promotion: Se True, promove a versão em Staging para Production
                              e arquiva as que estavam em Production.
        
        Returns:
            run_id do modelo
        """
        best_run_id, version, model_name = self.register_and_stage(
            metric=metric, model_name=model_name
        )
        
        if not approve_promotion:
            logger.info(
                "Promoção para Production não aprovada. Modelo permanece em Staging."
            )
            return best_run_id
        
        production_versions = self.client.get_latest_versions(
            model_name, stages=["Production"]
        )
        
        for pv in production_versions:
            self.client.transition_model_version_stage(
                name=model_name,
                version=pv.version,
                stage="Archived",
            )
            logger.info(
                f"Modelo '{model_name}' versão {pv.version} arquivado (era Production)"
            )
        
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
        )
        logger.info(
            f"Modelo '{model_name}' versão {version} promovido para Production"
        )
        
        return best_run_id