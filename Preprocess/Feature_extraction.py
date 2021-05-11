import numpy as np
import pandas as pd
import librosa
import os
from glob import glob
from scipy.stats import norm, kurtosis, skew, entropy
import essentia
import essentia.standard as es

class FeatureExtracter():
    df = pd.DataFrame()
    def __init__(self, path, mfcc=True, entropy=False):
        self.path = path
        self.data_files = glob(path + "/*/*.wav")
        self.spectral_features = ['spectral_centroid', 'spectral_bandwidth','spectral_contrast', 'spectral_flatness',
                    'spectral_rolloff', 'zero_crossing_rate','poly_features', 'rms']
        self.mfcc = mfcc
        self.entropy = entropy
        self.spectral_columns = []
        self.lpc_columns = []
        if mfcc:
            self.spectral_features.append('mfcc')
        
        self.genres = {v:k for k,v in enumerate([i[54:] for i in glob(path + "/*")])}
        self.fields = ['mean', 'std', 'kurtosis', 'skewness', 'median', 'min', 'max']
        if entropy:
            self.fields.append('entropy')
        
        classes = list(self.genres)
        
    def getColumnNames(self):
        features = []
        for i in self.spectral_features:
            features.append(i)
            features.append(i + "_delta1")
            features.append(i + "_delta2")

        spectral_columns = [i+"_"+j for i in features for j in self.fields]
        
        lpc_columns = ["LPC_"+i for i in self.fields]

        return spectral_columns, lpc_columns
        
    def calculate_values(self, arr):
        mean = arr.mean()
        std = arr.std()
        kurt = kurtosis(arr.T)[0]
        skewness = skew(arr.T)[0]
        median = np.median(arr)
        min_val = arr.min()
        max_val = arr.max()
        values = list((mean, std, kurt, skewness, median, min_val, max_val))
        if self.entropy:
            ent = entropy(arr)
            values.append(ent)
        return values
        
    def extract(self):
        spectral_df = []
        for i in self.data_files:
            x, sr = librosa.load(i)
            line = []
            for k in self.spectral_features:
                if k == 'spectral_flatness':
                    arr = eval("librosa.feature.{}(x)".format(k))
                else:
                    arr = eval("librosa.feature.{}(x,sr)".format(k))
                arr_delta1 = librosa.feature.delta(arr, order=1)
                arr_delta2 = librosa.feature.delta(arr, order=2)
                line += self.calculate_values(arr)
                line += self.calculate_values(arr_delta1)
                line += self.calculate_values(arr_delta2)
            spectral_df.append(line.copy())

        lpc_df = []
        for i in self.data_files:
            line = []
            audio = es.MonoLoader(filename=i)()
            lpc = es.LPC()(audio)[0]
            mean = lpc.mean()
            std = lpc.std()
            kurt = kurtosis(lpc)
            skewness = skew(lpc)
            median = np.median(lpc)
            min_val = lpc.min()
            max_val = lpc.max()
            line += [mean, std, kurt, skewness, median, min_val, max_val]
            lpc_df.append(line)
        
        spectral_columns, lpc_columns = self.getColumnNames()    
    
    def save(self, path):
        self.df.to_pickle(path, protocol=4)
        
if __name__=='__main__':
    extracter = FeatureExtracter("../Data/genres_original", mfcc=True, entropy=False)
    df = extracter.extract()
    extracter.save("../Data/extracter.pkl")