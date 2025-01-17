import matplotlib.pyplot as plt
import pickle
import constants
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import auc
from scipy import interp
import numpy as np
from itertools import cycle
"""
The evaluate class performs all necessary evaluation procedures for a given network.
"""
class Evaluator:
    """
    Initialize class
    -----------------------------------------------------------
    model: Keras object for the network to be evaluated
    show: boolean to choose if the evaluated graphs are displayed
          on screen in addition being saved as pngs
    """
    def __init__(self, model, show=True):
        self.model = model
        self.show = show
        print(model.summary())

    """
    Plot epoch loss graph
    -----------------------------------------------------------
    histroy: the history object of the model saved during Training
    name: Name of the model being evaluated
    """
    def plot_accloss_graph(self, histroy, name):
        plt.plot(histroy.history['acc'])
        plt.plot(histroy.history['val_acc'])
        plt.title('{} Model Accuracy'.format(name))
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.savefig('{}{}_aegraph.png'.format(constants.FIGURES, name))
        if self.show:
            plt.show()
        plt.clf()
        
    """
    Plot Confusion matrix
    -----------------------------------------------------------
    y_true: true labels
    y_pred: predicted labels
    name: Name of the model being evaluated
    """
    def plot_cm(self, y_true, y_pred, name):
        labels = ['Deepfake', 'Real']
        cm = confusion_matrix(y_true, y_pred, labels)
        print(cm)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        fig.colorbar(cax)
        plt.title('Confusion matrix')
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('{}{}_cm.png'.format(constants.FIGURES, name))

        if self.show:
            plt.show()
        plt.clf()

    """
    Plot ROC Curve
    -----------------------------------------------------------
    y_true: true labels
    y_score: predicted probabilities
    name: Name of the model being evaluated
    """
    def plot_roc(self, y_true, y_score, name):
        lw = 2
        n_classes = 2
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Combine false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Get average and calculate AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        colors = cycle(['red', 'blue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig('{}{}_roc.png'.format(constants.FIGURES, name))
        if self.show:
            plt.show()
        plt.clf()

    """
    Predict test data
    -----------------------------------------------------------
    x, y: independent and dependent variables
    name: Name of the model being evaluated
    """
    def predict_test_data(self, x, y, name):
        # TODO Check size is correct and resize?
        y_true = []
        y_true_multi = []
        y_score = []
        y_pred = []

        start = time.time()
        predictions = self.model.predict(x)
        end = time.time()
        print('{} completed the predictions in {}s ({}ps)'.format(name, round((end - start),2), round(len(predictions)/(end - start),2)))
        count = 0
        for pred in range(len(predictions)):
            if predictions[pred][0] > predictions[pred][1]:
                label = 'Real'
            else:
                label = 'Deepfake'

            y_score.append([predictions[pred][0],predictions[pred][1]])

            if y[pred][0] > y[pred][1]:
                true = 'Real'
                y_true_multi.append([1,0])
            else:
                true = 'Deepfake'
                y_true_multi.append([0,1])

            y_true.append(true)
            y_pred.append(label)

            if label is true:
                count += 1

        accuracy = count/len(predictions)
        self.plot_cm(y_true, y_pred, name)
        self.plot_roc(np.array(y_true_multi), np.array(y_score), name)
        print('Accuracy: {}%'.format(round((accuracy*100), 2)))

    """
    Set model
    -----------------------------------------------------------
    Sets the model to be evaluated
    model: New model Keras object
    """
    def set_model(self, model):
        self.model = model

    """
    Set show
    -----------------------------------------------------------
    Sets the figures to display in the GUI
    show: boolean for showing figures
    """
    def set_show(self, show):
        self.show = show

    """
    Get model
    -----------------------------------------------------------
    Returns model being evaluated
    """
    def get_model(self):
        return self.model
