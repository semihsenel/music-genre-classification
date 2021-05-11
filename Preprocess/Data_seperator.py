import numpy as np
import pandas as pd

if __name__ == '__main__':

	full_data = pd.read_pickle("../Data/extracter.pkl")
	full_data['label'] = [int(i//100) for i in range(0,1000)]

	columns = list(full_data.columns)

	spectral_features_with_delta = full_data[[i for i in columns if "spectral" in i or i == "label"]]
	spectral_features_without_delta = full_data[[i for i in columns if ("spectral" in i and "delta" not in i) or i == "label"]]
	mfcc_features_with_delta = full_data[[i for i in columns if "mfcc" in i or i == "label"]]
	mfcc_features_without_delta = full_data[[i for i in columns if ("mfcc" in i and "delta" not in i) or i == "label"]]
	lpc_features = full_data[[i for i in columns if "LPC" in i or i == "label"]]
	spectral_features_with_delta.to_csv("../Data/spectral_features_with_delta.csv", index=False)
	spectral_features_without_delta.to_csv("../Data/spectral_features_without_delta.csv", index=False)
	mfcc_features_with_delta.to_csv("../Data/mfcc_features_with_delta.csv", index=False)
	mfcc_features_without_delta.to_csv("../Data/mfcc_features_without_delta.csv", index=False)
	lpc_features.to_csv("../Data/lpc_features.csv", index=False)
	full_data = full_data.to_csv("../Data/full_data.csv", index=False)