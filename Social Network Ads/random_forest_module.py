
import numpy as np
import pandas as pd
import pickle





class naive_bayes:
    def __init__(self, model_file, scaler_file):
        with open('naive_bayes_model', 'rb') as model_file, open('scaler', 'rb') as scaler_file:
            self.classifier = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
            
    def load_and_scale_data(self, data_file):
        dataset = pd.read_csv(data_file).values
        scaled_dataset = self.scaler.transform(dataset)
        self.data = scaled_dataset
        
        
    def predict(self):
        if (self.data is not None):
            pred = self.classifier.predict(self.data)
            return pred
        
        

