import constants
import pickle

class Classifier:

    def __init__(self, name):
        self.model = self.load_model(name) #Load model

    def load_model(self, name):
        return pickle.load(open(constants.SAVED_MODELS + name + '.sav', 'rb'))

    def classify_video(self, video):
        pass
