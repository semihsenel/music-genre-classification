import numpy as np
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score


class ModelSelection():
	all_models = []
	def __init__(self, df, k, random_forest=True, logistic_reg=True, adaboost=True, 
				 decision_tree=True, knn=True, svm=True, naive_bayes=True, mlp=True, LDA=True):
		self.df = df
		self.k = k
		self.random_forest = random_forest
		self.logistic_reg = logistic_reg
		self.adaboost = adaboost
		self.decision_tree = decision_tree
		self.knn = knn
		self.svm = svm
		self.naive_bayes = naive_bayes
		self.mlp = mlp
		self.LDA = LDA
		
	def create_models(self):
		if self.random_forest:
			self.all_models.append(RandomForestClassifier(n_estimators=100, max_depth=10, max_features=0.6, min_samples_split=2))
			self.all_models.append(RandomForestClassifier(n_estimators=100, max_depth=20, max_features=0.6, min_samples_split=2))
			self.all_models.append(RandomForestClassifier(n_estimators=100, max_depth=10, max_features='sqrt', min_samples_split=2))
			self.all_models.append(RandomForestClassifier(n_estimators=100, max_depth=20, max_features='sqrt', min_samples_split=2))
			self.all_models.append(RandomForestClassifier(n_estimators=100, max_depth=50, max_features='auto', min_samples_split=2))

		if self.logistic_reg:
			self.all_models.append(LogisticRegression(C=10, max_iter=500, n_jobs=-1))
			self.all_models.append(LogisticRegression(C=1, max_iter=500, n_jobs=-1))
			self.all_models.append(LogisticRegression(C=0.1, max_iter=500, n_jobs=-1))
			self.all_models.append(LogisticRegression(C=0.01, max_iter=500, n_jobs=-1))
			self.all_models.append(LogisticRegression(C=0.001, max_iter=500, n_jobs=-1))

		if self.adaboost:
			self.all_models.append(AdaBoostClassifier(n_estimators=10, learning_rate=1, algorithm='SAMME.R'))
			self.all_models.append(AdaBoostClassifier(n_estimators=20, learning_rate=0.1, algorithm='SAMME.R'))
			self.all_models.append(AdaBoostClassifier(n_estimators=10, learning_rate=0.01, algorithm='SAMME.R'))
			self.all_models.append(AdaBoostClassifier(n_estimators=20, learning_rate=0.1, algorithm='SAMME'))
			self.all_models.append(AdaBoostClassifier(n_estimators=5, learning_rate=0.01, algorithm='SAMME'))

		if self.decision_tree:
			self.all_models.append(DecisionTreeClassifier(criterion='gini', max_depth=10))
			self.all_models.append(DecisionTreeClassifier(criterion='gini', max_depth=30))
			self.all_models.append(DecisionTreeClassifier(criterion='entropy', max_depth=10))
			self.all_models.append(DecisionTreeClassifier(criterion='entropy', max_depth=30))
			self.all_models.append(DecisionTreeClassifier(criterion='gini', max_depth=50))

		if self.knn:
			self.all_models.append(KNeighborsClassifier(n_neighbors=1, weights='uniform'))
			self.all_models.append(KNeighborsClassifier(n_neighbors=1, weights='distance'))
			self.all_models.append(KNeighborsClassifier(n_neighbors=2, weights='distance'))
			self.all_models.append(KNeighborsClassifier(n_neighbors=3, weights='uniform'))
			self.all_models.append(KNeighborsClassifier(n_neighbors=3, weights='distance'))

		if self.svm:
			self.all_models.append(SVC(kernel='linear'))
			self.all_models.append(SVC(kernel='poly', degree=3, C=1))
			self.all_models.append(SVC(kernel='poly', degree=4, C=1))
			self.all_models.append(SVC(kernel='poly', degree=5, C=1))
			self.all_models.append(SVC(kernel='rbf', degree=5, C=0.1))

		if self.mlp:
			self.all_models.append(MLPClassifier(hidden_layer_sizes=(100,100), solver='adam', learning_rate_init=0.001, max_iter=500, early_stopping=True))
			self.all_models.append(MLPClassifier(hidden_layer_sizes=(256,256), solver='adam', learning_rate_init=0.001, max_iter=500, early_stopping=True))
			self.all_models.append(MLPClassifier(hidden_layer_sizes=(100,100), solver='lbfgs', learning_rate_init=0.001, max_iter=500, early_stopping=True))
			self.all_models.append(MLPClassifier(hidden_layer_sizes=(256,256), solver='lbfgs', learning_rate_init=0.001, max_iter=1000, early_stopping=True))
			self.all_models.append(MLPClassifier(hidden_layer_sizes=(100,100), solver='lbfgs', learning_rate_init=1e-4, max_iter=1000, early_stopping=True))
		if self.LDA:

			self.all_models.append(LinearDiscriminantAnalysis())


	def apply_models(self):
		self.all_models.clear()
		self.create_models()
		X,y = self.split()
		scores = []
		for i in self.all_models:
			accuracy = cross_val_score(i,X,y,scoring='accuracy').mean()
			recall = cross_val_score(i,X,y,scoring='recall_macro').mean()
			precision = cross_val_score(i,X,y,scoring='precision_macro').mean()
			f1 = cross_val_score(i,X,y,scoring='f1_macro').mean()
			scores.append("\n{} ------>\nCross Validation Score (accuracy) : {}\nCross Validation Score (recall) : {}\nCross Validation Score (precision) : {}\nCross Validation Score (f1) : {}".format(i,accuracy, recall, precision, f1))
		return scores
	
	def apply_models_with_pca(self):
		self.all_models.clear()
		self.create_models()
		X,y = self.split()
		scores = []
		pca = PCA(n_components=0.99, svd_solver='full')
		new_X = pca.fit_transform(X)
		for i in self.all_models:
			accuracy = cross_val_score(i,X,y,scoring='accuracy').mean()
			recall = cross_val_score(i,X,y,scoring='recall_macro').mean()
			precision = cross_val_score(i,X,y,scoring='precision_macro').mean()
			f1 = cross_val_score(i,X,y,scoring='f1_macro').mean()
			scores.append("\n{} ------>\nCross Validation Score (accuracy) : {}\nCross Validation Score (recall) : {}\nCross Validation Score (precision) : {}\nCross Validation Score (f1) : {}".format(i,accuracy, recall, precision, f1))
		return scores
	
	def apply_models_with_feature_elimination(self):
		self.all_models.clear()
		self.create_models()
		estimator = SVC(kernel='linear')
		column_names = list(self.df.columns)
		column_names.remove('label')
		max_score = 0
		start = 100 if len(column_names) > 100 else 2
		true_drop_columns = []
		for i in range(100, len(column_names)):     
			X,y = self.split()
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
			selector = RFE(estimator, n_features_to_select=i, step=1, verbose=0)
			selector.fit(X_train, y_train)
			drop_columns = [column_names[i] for i in range(len(column_names)) if not selector.support_[i]]
			model = SVC(kernel="linear")
			X,y = self.split(self.df.drop(drop_columns, axis=1))
			score = cross_val_score(model,X,y).mean()
			if score > max_score:
				max_score = score
				true_drop_columns = drop_columns.copy()
		
		X,y = self.split(self.df.drop(true_drop_columns, axis=1))
		scores = []
		for i in self.all_models:
			accuracy = cross_val_score(i,X,y,scoring='accuracy').mean()
			recall = cross_val_score(i,X,y,scoring='recall_macro').mean()
			precision = cross_val_score(i,X,y,scoring='precision_macro').mean()
			f1 = cross_val_score(i,X,y,scoring='f1_macro').mean()
			scores.append("\n{} ------>\nCross Validation Score (accuracy) : {}\nCross Validation Score (recall) : {}\nCross Validation Score (precision) : {}\nCross Validation Score (f1) : {}".format(i,accuracy, recall, precision, f1))
		return scores, true_drop_columns

		
	def split(self, df = None):
		if df is None:
			df_copy = self.df.sample(frac=1).reset_index(drop=True)
			X = df_copy.drop('label',axis=1).values
			y = df_copy['label'].values
			sc = StandardScaler()
			X = sc.fit_transform(X)
			return X,y
		else:
			df_copy = df.sample(frac=1).reset_index(drop=True)
			X = df_copy.drop('label',axis=1).values
			y = df_copy['label'].values
			sc = StandardScaler()
			X = sc.fit_transform(X)
			return X,y

def writeFile(scores, filename):
	with open("../Outputs/{}.txt".format(filename),"w") as f:
		for i in scores:
			f.write(i)

if __name__ == '__main__':
	df = pd.read_csv("../Data/full_data.csv")
	model_selector = ModelSelection(df, 5, naive_bayes=False)
	
	pca_scores = model_selector.apply_models_with_pca()
	writeFile(pca_scores, "pca_scores")

	feature_elimination_scores, true_drop_columns = model_selector.apply_models_with_feature_elimination()
	writeFile(feature_elimination_scores, "feature_elimination_scores")

	normal_scores = model_selector.apply_models()
	writeFile(normal_scores, "normal_scores")

	just_lpc = pd.read_csv("../Data/lpc_features.csv")
	lpc_model_selector = ModelSelection(just_lpc, 5, naive_bayes=False)

	lpc_pca_scores = lpc_model_selector.apply_models_with_pca()
	writeFile(lpc_pca_scores, "lpc_pca_scores")

	lpc_feature_elimination_scores, lpc_true_drop_columns = lpc_model_selector.apply_models_with_feature_elimination()
	writeFile(lpc_feature_elimination_scores, "lpc_feature_elimination_scores")
	
	lpc_normal_scores = lpc_model_selector.apply_models()
	writeFile(lpc_normal_scores, "lpc_normal_scores")

	just_mfcc = pd.read_csv("../Data/mfcc_features_with_delta.csv")
	model_selector = ModelSelection(just_mfcc, 5, naive_bayes=False)

	mfcc_normal_scores = model_selector.apply_models()
	writeFile(mfcc_normal_scores, "mfcc_normal_scores")

	mfcc_pca_scores = model_selector.apply_models_with_pca()
	writeFile(mfcc_pca_scores, "mfcc_pca_scores")

	mfcc_feature_elimination_scores, mfcc_true_drop_columns= model_selector.apply_models_with_feature_elimination()
	writeFile(mfcc_feature_elimination_scores, "mfcc_feature_elimination_scores")
	
	
	
	
	
	
	
	
	