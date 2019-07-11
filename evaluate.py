import matplotlib.pyplot as plt
import pickle
import constants
import time

class Evaluator:
    def __init__(model):
        self.model = model

    def plot_accloss_graph(self, histroy, name):
        plt.plot(histroy.history['acc'])
        plt.plot(histroy.history['val_acc'])
        plt.title('{} Model Accuracy'.format(name))
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.show()

    def predict_test_data(self, x, y, name):
        start = time.time()
        predictions = self.model.predict(x)
        end = time.time()
        print('Predictions completed in {}s'.format(end - start))
        count = 0
        for pred in range(len(predictions)):
            if predictions[pred][0] > predictions[pred][1]:
                label = 1
            else:
                label = 0

            if y[pred][0] > y[pred][1]:
                true = 1
            else:
                true = 0

            if label is true:
                count += 1

        accuracy = count/len(predictions)
        print(accuracy)
