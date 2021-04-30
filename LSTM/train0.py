from LSTM.driving_alert_lstm import DrivingAlertLSTM
from LSTM.training_model import TrainingModel


def main():
    model = TrainingModel(new_model_func=DrivingAlertLSTM)
    model.training()


if __name__ == '__main__':
    main()
