import trainer
import models


def main(dataset_path):
    trainer.train(dataset_path, [ models.bi_lstm_model])


if __name__ == '__main__':
    main(dataset_path='datasets/spam_ham_v2.csv')
