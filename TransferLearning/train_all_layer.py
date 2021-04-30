from TransferLearning.training_model import TrainingModel, new_model


def main():
    model = TrainingModel(new_model_func=lambda: new_model(train_or_freeze=False, target_layer=[]), env_name='all')
    model.training()


if __name__ == '__main__':
    main()
