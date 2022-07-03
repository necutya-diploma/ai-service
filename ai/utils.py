from typing import Any

import pandas as pd
from matplotlib import pyplot as plt


def visualize_train_result( model_name: str, history: Any):
    # Read as a dataframe
    metrics = pd.DataFrame(history.history)

    # Rename column
    metrics.rename(
        columns={'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy',
                 'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'},
        inplace=True,
    )

    def plot_graphs1(var1, var2, string, model_name):
        metrics[[var1, var2]].plot()
        plt.title(f'{model_name}: training and Validation ' + string)
        plt.xlabel('Number of epochs')
        plt.ylabel(string)
        plt.legend([var1, var2])
        print(metrics[[var1]])
        print(metrics[[var2]])
        print(metrics[[var1, var2]])
        plt.show()

    plot_graphs1('Training_Loss', 'Validation_Loss', 'loss', model_name)
    plot_graphs1('Training_Accuracy', 'Validation_Accuracy', 'accuracy', model_name)
