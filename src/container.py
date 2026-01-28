from dependency_injector import containers, providers
from src.data.ingestion import Ingestion

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