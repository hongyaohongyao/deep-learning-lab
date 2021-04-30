from LSTM.driving_alert_lstm import DrivingAlertLSTM
from LSTM.training_model import TrainingModel


def main():
    model = TrainingModel(new_model_func=lambda: DrivingAlertLSTM(bidirectional=True), env_name='bi')
    model.training()


if __name__ == '__main__':
    main()
