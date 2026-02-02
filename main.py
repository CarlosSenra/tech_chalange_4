from src.container import Container

container = Container()
container.config.from_yaml('config/config.yml')

if __name__ == '__main__':
    if container.config.state.train:
        train = container.train()
        train.run()
    else:
        print("O estado de treinamento está desativado")