import numpy as np
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.base import BaseEstimator, ClassifierMixin

def readout_fidelity(y_pred, y_true):
	false_negative_rate = np.sum(y_true*(1-y_pred))/np.sum(y_true)
	false_positive_rate = np.sum((1-y_true)*y_pred)/np.sum(1-y_true)
	return 1-(false_negative_rate+false_positive_rate)/2
	
readout_fidelity_scorer = make_scorer(readout_fidelity)

def evaluate_classifier(classifier, X, y, return_test_score_error=False, return_train_score=False, return_train_score_error=False):
	scores = cross_validate(classifier, X, y, scoring={'fidelity':readout_fidelity_scorer, 'roc_auc': make_scorer(roc_auc_score, needs_proba=True)},
							 cv=3, return_train_score=(return_train_score or return_train_score_error) )
#    if (return_train_score or return_train_score_error):
#		train_scores = scores['train_score']
	print(scores)
	fidelity  = scores['test_fidelity']
	roc_auc  = scores['test_roc_auc']
	#return np.mean(test_scores), np.std(test_scores)/np.sqrt(len(test_scores)),\
	#        np.mean(train_scores), np.std(train_scores)/np.sqrt(len(train_scores))
#	retvals = [np.mean(test_scores)]
#	if return_test_score:
#	retvals.append()
	return {'fidelity':np.mean(fidelity), 'roc_auc':np.mean(roc_auc)}
	
class linear_classifier(BaseEstimator, ClassifierMixin):
	def __init__(self):
		pass
	# def feature_reducer(source, src_meas, axis_mean, bg, feature):
		# def get_points():
			# new_axes = source.get_points()[src_meas].copy()
			# del new_axes [axis_mean]
			# return new_axes

# +			
		# self.filter = self.predictor_class.dimreduce
		# self.filter_binary = self.predictor_class.predict
		# self.filter_mean = lambda x: self.predictor_class.predict.
		# pass
		
	def fit(self, X, y):
		self.avg_zero = np.mean(X[y==0, :], axis=0)
		self.avg_one  = np.mean(X[y==1, :], axis=0)
		
		self.avg = (self.avg_zero+self.avg_one)/2
		self.diff = (self.avg_one-self.avg_zero)/2
		self.sigma_0 = np.dot(np.conj(X[y==0,:]-self.avg_zero).T, X[y==0,:]-self.avg_zero)
		self.sigma_1 = np.dot(np.conj(X[y==1,:]-self.avg_one).T, X[y==1,:]-self.avg_one)
		self.Sigma_inv = np.linalg.inv(self.sigma_0+self.sigma_1)
		#self.Sigma_inv = np.identity(len(self.avg))
		#self.Sigma_inv = np.linalg.inv(np.dot(np.conj(X-self.avg).T, X-self.avg))
		#print (self.Sigma_inv)
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
		