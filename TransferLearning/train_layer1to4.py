from TransferLearning.training_model import TrainingModel, new_model


def main():
    model = TrainingModel(new_model_func=new_model, env_name='layer1to4')
    model.training()


if __name__ == '__main__':
    main()
