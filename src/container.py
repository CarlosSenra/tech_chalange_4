from dependency_injector import containers, providers
from src.data.ingestion import Ingestion
from src.data.feature_eng import FeatureEng

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