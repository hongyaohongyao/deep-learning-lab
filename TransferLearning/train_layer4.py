from TransferLearning.training_model import TrainingModel, new_model


def main():
    model = TrainingModel(new_model_func=lambda: new_model(target_layer=['layer4']), env_name='layer4')
    model.training()


if __name__ == '__main__':
    main()
