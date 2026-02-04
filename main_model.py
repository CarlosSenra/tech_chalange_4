from src.container import Container

container = Container()
container.config.from_yaml('config/config.yml')

train_state = True
evaluate_state = False

if __name__ == '__main__':
    if train_state:
        train = container.train()
        train.run()
    else:
        print("O estado de treinamento está desativado")

    if evaluate_state:
        evaluate = container.evaluate()
        eval = evaluate.get_best_model(metric="val_mae")
        print(eval)
        promote = evaluate.promote_to_production(metric="val_mae")
        print(promote)
    else:
        print("O estado de avaliação está desativado")


    