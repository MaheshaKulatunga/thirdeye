import matplotlib.pyplot as plt
import pickle
import constants

def plot_accloss_graph(histroy, name):
    plt.plot(histroy.history['acc'])
    plt.plot(histroy.history['val_acc'])
    plt.title('{} Model Accuracy'.format(name))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()
