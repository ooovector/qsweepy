import numpy as np
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.base import BaseEstimator, ClassifierMixin

def binary_readout_fidelity(y_pred, y_true):
	false_negative_rate = np.sum(y_true*(1-y_pred))/np.sum(y_true)
	false_positive_rate = np.sum((1-y_true)*y_pred)/np.sum(1-y_true)
	return 1-(false_negative_rate+false_positive_rate)/2
	
binary_readout_fidelity_scorer = make_scorer(binary_readout_fidelity)

def binary_evaluate_classifier(classifier, X, y):
	scores = cross_validate(classifier, X, y, scoring={'fidelity':binary_readout_fidelity_scorer, 'roc_auc': make_scorer(roc_auc_score, needs_proba=True)},
							 cv=3, return_train_score=(return_train_score or return_train_score_error) )
	fidelity  = scores['test_fidelity']
	roc_auc  = scores['test_roc_auc']
	return {'fidelity':np.mean(fidelity), 'roc_auc':np.mean(roc_auc)}
	
def readout_fidelity(y_pred, y_true):
	return np.sum(y_pred == y_true)/len(y_true)

readout_fidelity_scorer = make_scorer(readout_fidelity)
	
def confusion_matrix(y_pred_proba, y_true):
	'''
	Assignment probability of single-shot readout.
	
	y_pred_proba(numpy.ndarray, shape (M, N)), M -- number of samples, N -- number of classes
	returns: (N, N) numpy.ndarray assignment
	'''
	confusion_matrix = np.zeros(y_pred_proba.shape[1], y_pred_proba.shape[1])
	for _class_id in range(y_pred_proba.shape[1]):
		confusion_matrix[_class_id,:] = np.mean(y_pred_proba[y_true==_class_id], axis=0)
	return confusion_matrix
	
def probability_aware_readout_fidelity(y_pred_proba, y_true):
	return np.trace(confusion_matrix(y_pred_proba, y_true))

probability_aware_readout_fidelity_scorer = make_scorer(probability_aware_readout_fidelity, needs_proba=True)
	
def evaluate_classifier(classifier, X, y):
	scores = cross_validate(classifier, X, y, scoring={'fidelity':readout_fidelity_scorer, 'probability_aware_fidelity': probability_aware_readout_fidelity_scorer},
							 cv=3, return_train_score=(return_train_score or return_train_score_error) )
	fidelity  = scores['test_fidelity']
	roc_auc  = scores['test_probability_aware_fidelity']
	return {'fidelity':np.mean(fidelity), 'probability_aware_fidelity':np.mean(roc_auc)}	

# class linear_classifier(BaseEstimator, ClassifierMixin):
	# def __init__(self):
		# self.purify = True
		# pass

	# def fit(self, X, y):
		# self.class_averages = {}
		# for _class_id in list(set(y)):
			# self.class_averages[_class_id] = np.mean(X[y==_class_id,:], axis=0)
			# dev = X[y==_class_id,:]-self.class_averages[_class_id].T
			# self.class_cov[_class_id] = np.dot(np.conj(dev.T), dev)
		# self.
			
	
class binary_linear_classifier(BaseEstimator, ClassifierMixin):
	def __init__(self):
		pass
		
	def fit(self, X, y):
		self.avg_zero = np.mean(X[y==0, :], axis=0)
		self.avg_one  = np.mean(X[y==1, :], axis=0)
		
		self.avg = (self.avg_zero+self.avg_one)/2
		self.diff = (self.avg_one-self.avg_zero)/2
		self.sigma_0 = np.dot(np.conj(X[y==0,:]-self.avg_zero).T, X[y==0,:]-self.avg_zero)
		self.sigma_1 = np.dot(np.conj(X[y==1,:]-self.avg_one).T, X[y==1,:]-self.avg_one)
		self.Sigma_inv = np.linalg.inv(self.sigma_0+self.sigma_1)
		self.feature = self.Sigma_inv
		self.naive_bayes(X,y)
		
	def naive_bayes(self, X, y):
		predictions = self.dimreduce(X)
		
		hist_all, bins = np.histogram(predictions, bins='auto')
		proba_points = (bins[1:]+bins[:-1])/2.
		hists = []
		
		for y_val in range(2):
			hists.append(np.histogram(predictions[y==y_val], bins=bins)[0])
			
		hists = np.asarray(hists, dtype=float)
		probabilities = hists/hist_all
		naive_probabilities = np.asarray([proba_points<0, proba_points>0], dtype=float)
		probabilities[np.isnan(probabilities)] = naive_probabilities[np.isnan(probabilities)]
		self.probabilities = probabilities
		self.proba_points = proba_points
		self.hists = hists
	def predict_proba(self, X):
		return np.interp(self.dimreduce(X), self.proba_points, self.probabilities[1,:], left=0., right=1.)
	
	def dimreduce(self, X):
		prediction = np.real(np.sum(np.dot(np.conj(self.diff),self.Sigma_inv)*(X - self.avg), axis=1))
		return prediction
	def predict(self, X):
		prediction = self.dimreduce(X)>0
		return prediction
		