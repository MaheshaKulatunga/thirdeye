import matplotlib.pyplot as plt
import pickle
import constants

if __name__ == "__main__":
    # providence = pickle.load(open(constants.SAVED_MODELS + 'providence.sav', 'rb'))
    print('Evaluating model')
    providence_history = pickle.load(open(constants.SAVED_MODELS + 'providence_history.sav', 'rb'))
    plt.plot(providence_history.history['acc'])
    plt.plot(providence_history.history['val_acc'])
    plt.title('Providence Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()
