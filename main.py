from src.container import Container

container = Container()
container.config.from_yaml('config/config.yml')

if __name__ == '__main__':
    ingestion = container.ingestion()
    ingestion.save_raw_data()