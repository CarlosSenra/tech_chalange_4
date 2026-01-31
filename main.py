from src.container import Container

container = Container()
container.config.from_yaml('config/config.yml')

if __name__ == '__main__':
    if False:
        ingestion = container.ingestion()
        ingestion.save_raw_data()

        print(container.config.data.feature_eng.features_X())

        feature_eng = container.feature_eng()
        x_train, y_train, x_test, y_test = feature_eng.run(janela=5)
        print(x_train[:5])
        print(y_train[:5])
        print(x_test[:5])
        print(y_test[:5])
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)
        
        x_train, y_train = feature_eng.reverse_sequences(x_train, y_train)
        x_test, y_test = feature_eng.reverse_sequences(x_test, y_test)
        print(x_train[:5])
        print(y_train[:5])
        print(x_test[:5])
        print(y_test[:5])
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)

    train = container.train()
    train.run()