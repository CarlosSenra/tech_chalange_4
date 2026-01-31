from dependency_injector import containers, providers
from src.data.ingestion import Ingestion
from src.data.feature_eng import FeatureEng
from src.models.LSTM.SimpleLSTM import SimpleLSTM
from src.models.train.train import Train

class Container(containers.DeclarativeContainer):

    config = providers.Configuration()

    ingestion = providers.Factory(
        Ingestion,
        ticker=config.data.ingestion.ticker,
        range_period=config.data.ingestion.range_period,
        interval=config.data.ingestion.interval,
        raw_path=config.data.path.raw,
        staging_path=config.data.path.staging
    )

    feature_eng = providers.Factory(
        FeatureEng,
        raw_path=config.data.path.raw,
        ticker=config.data.ingestion.ticker,
        test_size=config.data.feature_eng.test_size,
        features_X=config.data.feature_eng.features_X,
        features_y=config.data.feature_eng.features_y
    )

    simple_lstm = providers.Singleton(
        SimpleLSTM,
        list_units=config.models.lstm.SimpleLSTM.list_units,
        output_dim=config.models.lstm.SimpleLSTM.output_dim
    )

    train = providers.Factory(
        Train,
        model=simple_lstm,
        feature_eng=feature_eng,
        experiment_name=config.models.train.experiment_name,
        epochs=config.models.train.epochs,
        batch_size=config.models.train.batch_size,
        validation_split=config.models.train.validation_split,
        verbose=config.models.train.verbose,
        metric_list=config.models.train.metric_list,
        optimizer=config.models.train.optimizer,
        loss=config.models.train.loss
    )