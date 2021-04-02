import numpy as np
import matplotlib.pyplot as plt

from train import *


def predict(path='./Data', saved_path='./saved_models/', index=0, display_performance=True):
    test_df = get_test_df(path)
    # print("data_loaded")

    classifier_model, histories = train()

    results = classifier_model.predict(test_df['text'])

    threshold = 0.5
    results = np.where(results > threshold, 1, 0)

    if display_performance is True:
        history_dict = histories[get_model_name(index)].history
        acc = history_dict['binary_accuracy']
        val_acc = history_dict['val_binary_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(acc) + 1)
        fig = plt.figure(figsize=(10, 6))
        fig.tight_layout()

        plt.subplot(2, 1, 1)
        # "bo" is for "blue dot"
        plt.plot(epochs, loss, 'r', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        # plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

    return results


if __name__ == "__main__":
    predict()
