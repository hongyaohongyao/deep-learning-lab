from LSTM.driving_alert_lstm import DrivingAlertLSTM
from LSTM.training_model import TrainingModel


def main():
    model = TrainingModel(new_model_func=lambda: DrivingAlertLSTM(num_layers=4), env_name='4layer')
    model.training()


if __name__ == '__main__':
    main()
