import constants
import pickle

class Classifier:

    def __init__(self, model):
        self.model = model

    def classify_video(self, video):
        prediction = self.model.predict([video])
        return prediction
