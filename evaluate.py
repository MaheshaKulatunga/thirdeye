import matplotlib.pyplot as plt
import pickle
import constants
import time

class Evaluator:
    def __init__(self, model):
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
        # TODO Check size is correct and resize?
        start = time.time()
        predictions = self.model.predict(x)
        end = time.time()
        print('{} completed the predictions in {}s'.format(name, (end - start)))
        count = 0
        for pred in range(len(predictions)):
            if predictions[pred][0] > predictions[pred][1]:
                label = 0
            else:
                label = 1

            if y[pred][0] > y[pred][1]:
                true = 0
            else:
                true = 1

            if label is true:
                count += 1

        accuracy = count/len(predictions)
        print(accuracy)

    def set_model(self, model):
        self.model = model

    def get_model(self, model):
        return model
