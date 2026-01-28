from src.container import Container

container = Container()
container.config.from_yaml('config/config.yml')

if __name__ == '__main__':
    ingestion = container.ingestion()
    ingestion.save_raw_data()

    print(container.config.data.feature_eng.features_X())

    feature_eng = container.feature_eng()
    x, y = feature_eng.create_sequences(janela=5)
    print(x[:5])
    print(y[:5])
    print(x.shape)
    print(y.shape)
    x, y = feature_eng.reverse_sequences(x, y)
    
