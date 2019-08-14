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
	scores = cross_validate(classifier, X, y, scoring={'fidelity':binary_readout_fidelity_scorer, 'roc_auc': make_scorer(roc_auc_score, needs_proba=True)}, cv=3)
	fidelity  = scores['test_fidelity']
	roc_auc  = scores['test_roc_auc']
	return {'fidelity':np.mean(fidelity), 'roc_auc':np.mean(roc_auc)}

def readout_fidelity(y_true, y_pred):
	return np.sum(y_pred == y_true)/len(y_true)

readout_fidelity_scorer = make_scorer(readout_fidelity)

def confusion_matrix(y_true, y_pred):
	'''
	Assignment probability of single-shot readout.

	y_pred_proba(numpy.ndarray, shape (M, N)), M -- number of samples, N -- number of classes
	returns: (N, N) numpy.ndarray assignment
	'''
	#print(y_pred_proba.shape)
	#print (y_true.tolist())
	#print (y_pred.tolist())
	#print (set(t_true.tolist()))
	confusion_matrix = np.zeros((len(set(y_true.tolist())), len(set(y_pred.tolist()))))
	for true_class_id, true_class in enumerate(sorted(list(set(y_true)))):
		for pred_class_id, pred_class in enumerate(sorted(list(set(y_pred)))):
			confusion_matrix[true_class_id,pred_class_id] = np.mean(np.logical_and(y_true==true_class,y_pred==pred_class), axis=0)/np.mean(y_true==true_class)
	#print (confusion_matrix)
	return confusion_matrix

def confusion_matrix_with_proba(y_true, y_pred_proba):
	'''
	Assignment probability of single-shot readout.

	y_pred_proba(numpy.ndarray, shape (M, N)), M -- number of samples, N -- number of classes
	returns: (N, N) numpy.ndarray assignment
	'''
	#print(y_pred_proba.shape)
	confusion_matrix = np.zeros((y_pred_proba.shape[1], y_pred_proba.shape[1]))
	for _class_id in range(y_pred_proba.shape[1]):
		confusion_matrix[_class_id,:] = np.mean(y_pred_proba[y_true==_class_id,:], axis=0)
	#print (confusion_matrix)
	return confusion_matrix

def probability_aware_readout_fidelity(y_true, y_pred_proba):
	return np.trace(confusion_matrix_with_proba(y_true, y_pred_proba))/y_pred_proba.shape[1]

probability_aware_readout_fidelity_scorer = make_scorer(probability_aware_readout_fidelity, needs_proba=True)

def evaluate_classifier(classifier, X, y):
	scores = cross_validate(classifier, X, y, scoring={'fidelity':readout_fidelity_scorer})#, 'probability_aware_fidelity': probability_aware_readout_fidelity_scorer}, cv=3)
	fidelity  = scores['test_fidelity']
	#roc_auc  = scores['test_probability_aware_fidelity']
	return {'fidelity':np.mean(fidelity)}#, 'probability_aware_fidelity':np.mean(roc_auc)}

readout_classifier_scores = ['fidelity']#, 'probability_aware_fidelity']

class linear_classifier(BaseEstimator, ClassifierMixin):
	def __init__(self, purify=True, cov_mode='equal'):
		self.purify = purify
		self.cov_mode = cov_mode
		self.class_list = [0, 1]
		self.nbins = 20
		pass

	def fit(self, X, y):
		self.class_averages = {}
		self.class_cov = {}
		self.class_list = sorted(list(set(y)))
		# go noavg
		#_X = X.copy()
		X = X - np.reshape(np.mean(X, axis=1), (-1, 1))
		for _class_id in self.class_list:
			self.class_averages[_class_id] = np.mean(X[y==_class_id,:], axis=0)
		if self.cov_mode in ['LDA', 'QDA']:
			for _class_id in self.class_list:
				dev = X[y==_class_id,:]-self.class_averages[_class_id].T
				self.class_cov[_class_id] = np.dot(np.conj(dev.T), dev)/(np.sum(y==_class_id)-1)
			#self.class_cov[_class_id] = np.cov(dev, rowvar=False)
		elif self.cov_mode == 'equal':
			self.cov_inv = 1./np.mean([np.std(np.abs(X[y==_class_id,:]-self.class_averages[_class_id].T), axis=0)**2 for _class_id in self.class_list])
			self.class_features = {_class_id: np.conj(self.class_averages[_class_id])*self.cov_inv for _class_id in self.class_list}
		if self.cov_mode == 'LDA':
			self.cov_inv = np.linalg.inv(np.sum([c for c in self.class_cov.values()],axis=0))
			self.class_cov_inv = {_class_id: self.cov_inv for _class_id in self.class_list}
			self.class_features = {_class_id: np.dot(np.conj(self.class_averages[_class_id]), self.cov_inv.T) for _class_id in self.class_list}
		elif self.cov_mode == 'QDA':
			self.class_cov_inv = {_class_id: np.linalg.inv(self.class_cov[_class_id]) for _class_id in self.class_list}
			## TODO: insert correct formula for QDA here
			#self.class_features = {_class_id: np.dot(self.class_cov[_class_id], self.cov_inv)/len(self.class_cov[_class_id]) for _class_id in list(set(y))}
		self.class_features = {_class_id: feature - np.mean(feature) for _class_id,feature in self.class_features.items()}
		self.naive_bayes(X, y)

	def dimreduce(self, X):
		#_X = X.copy()
		#X = X - np.reshape(np.mean(X, axis=1), (-1, 1))
		if self.cov_mode in ['equal', 'LDA']:
			reduced = [np.sum(np.real(self.class_features[_class_id]*X), axis=1) for _class_id in self.class_list]
		#prediction = np.real(np.sum(np.dot(np.conj(self.diff),self.Sigma_inv)*(X - self.avg), axis=1))
		#print (np.asarray(reduced).shape)
		return reduced

	def naive_bayes(self, X, y):
		from scipy.interpolate import griddata
		predictions = np.asarray(self.dimreduce(X))
		# reduce last class dimension
		predictions = np.asarray(predictions - np.mean(predictions, axis=0))[:-1, :]

		hist_all, bins = np.histogramdd(predictions.T, bins=self.nbins)
		proba_points = [(d_bins[1:]+d_bins[:-1])/2. for d_bins in bins]
		hists = []

		for y_val in self.class_list:
			hists.append(np.histogramdd(predictions[:,y==y_val].T, bins=bins)[0])

		hists = np.asarray(hists, dtype=float)
		probabilities = hists/hist_all
		points = np.reshape(np.meshgrid(*proba_points), (predictions.shape[0], -1)).T
		#naive_probabilities = np.asarray([proba_points<0, proba_points>0], dtype=float)
		#probabilities[np.isnan(probabilities)] = naive_probabilities[np.isnan(probabilities)]
		#X = np.asarray(np.meshgrid(*tuple(bins)))
		#probabilities[np.isnan(probabilities)] = self.predict_by_nearest([_x[np.isnan(probabilities)] for _x in X])
		#probability_nearest =
		#print (probabilities[].shape)
		nonzero_points = np.asarray([points[hist_all.ravel()!=0, p_id] for p_id in range(points.shape[1])])
		zero_points = np.asarray([points[hist_all.ravel()==0, p_id] for p_id in range(points.shape[1])])
		#nonzero_0class_prob = probabilities[0,...][hist_all!=0]
		#print(probabilities[0, ...].shape, nonzero_points.shape, zero_points.shape, nonzero_0class_prob.shape)
		unnormalized_nearest = [griddata(nonzero_points.T,
										 probabilities[y_val_id,...][hist_all!=0],
										 zero_points.T,
										 method='nearest') for y_val_id, y_val in enumerate(self.class_list)]
		#print ('predictions:', predictions)
		#print ('points:', points, 'probabilities:', 	probabilities)
		#print ('probabilities[0]:', probabilities[0,...] )
		#print ('hist_all:', hist_all, 'bins:', bins)
		#print ('unnormalized_nearest:', unnormalized_nearest)
		#print(nonzero_0class_prob)
		for y_val_id, y_val in enumerate(self.class_list):
		#	if len(unnormalized_nearest[y_val_id]):
				probabilities[y_val_id,...][hist_all==0] = (unnormalized_nearest[y_val_id] / np.sum(unnormalized_nearest, axis=0)).ravel()

		self.probabilities = probabilities
		self.proba_points = tuple(proba_points)
		self.hists = hists

	def predict(self, X):
		#return np.interp(self.dimreduce(X), self.proba_points, self.probabilities[1,:], left=0., right=1.)
		return self.predict_by_nearest(X)

	def predict_by_nearest(self, X):
		result = np.vectorize(self.class_list.__getitem__)(np.argmax(self.dimreduce(X), axis=0))
		#print('predict clled, returned shape:', result.shape)
		return result
		#print (np.argmax(self.dimreduce(X), axis=0).shape)
		#return self.class_list[np.argmax(self.dimreduce(X), axis=0)]
	# trivial probability assignment
	def predict_proba(self, X):
		#result = np.zeros((np.asarray(X).shape[0], len(self.class_list)))
		#indeces = np.arange(np.asarray(X).shape[0])
		#result[np.asarray((indeces, self.predict(X))).T]=1.
		#print('predict_proba called, returned shape:', result.shape, 'set 1 shape', np.asarray((indeces, self.predict(X))).shape)
		#from scipy.sparse import coo_matrix
		#result = coo_matrix((np.ones(np.asarray(X).shape[0]), ((np.arange(np.asarray(X).shape[0]), self.predict(X)))), (np.asarray(X).shape[0], len(self.class_list)))
		#return result.todense()
		from scipy.interpolate import interpn
		predictions = np.asarray(self.dimreduce(X))
		# reduce last class dimension
		predictions = np.asarray(predictions - np.mean(predictions, axis=0))[:-1, :]
		#print (len(self.proba_points))
		#print (self.probabilities.shape)
		result = np.asarray([interpn(self.proba_points, self.probabilities[_class_id,...], np.asarray(predictions).T, method='nearest', bounds_error=False, fill_value=None) for _class_id, class_name in enumerate(self.class_list)]).T
		#print (result.shape)
		return result




class binary_linear_classifier(BaseEstimator, ClassifierMixin):
	def __init__(self):
		self.nbins=20
		self.class_list = [0, 1]
		pass

	def fit(self, X, y):
		self.avg_zero = np.mean(X[y==0, :], axis=0)
		self.avg_one  = np.mean(X[y==1, :], axis=0)
		self.class_averages = {0:self.avg_zero, 1:self.avg_one}

		self.avg = (self.avg_zero+self.avg_one)/2
		self.diff = (self.avg_one-self.avg_zero)/2
		#self.sigma_0 = np.dot(np.conj(X[y==0,:]-self.avg_zero).T, X[y==0,:]-self.avg_zero)
		#self.sigma_1 = np.dot(np.conj(X[y==1,:]-self.avg_one).T, X[y==1,:]-self.avg_one)
		#self.Sigma_inv = np.linalg.inv(self.sigma_0+self.sigma_1)
		#self.feature = self.Sigma_inv
		self.feature = self.diff - np.mean(self.diff)#self.diff-np.mean(self.diff)
		self.naive_bayes(X,y)

	def naive_bayes(self, X, y):
		from matplotlib.pyplot import plot, axvline, figure
		predictions = self.dimreduce(X)

		hist_all, bins = np.histogram(predictions, bins=self.nbins)
		proba_points = (bins[1:]+bins[:-1])/2.
		hists = []

		for y_val in range(2):
			hists.append(np.histogram(predictions[y==y_val], bins=bins)[0])

		hists = np.asarray(hists, dtype=float)
		probabilities = hists/hist_all
		naive_probabilities = np.asarray([proba_points<0, proba_points>0], dtype=float)
		probabilities[np.isnan(probabilities)] = naive_probabilities[np.isnan(probabilities)]
		cdf = np.cumsum(hists, axis=1)
		self.threshold = proba_points[np.argmax((np.max(cdf)-cdf[0,:]-cdf[1,:])<0)]
		self.probabilities = probabilities
		self.proba_points = proba_points
		self.hists = hists

	def predict_proba(self, X):
		return np.reshape(np.interp(self.dimreduce(X), self.proba_points, self.probabilities[1,:], left=0., right=1.), (-1, 1))

	def dimreduce(self, X):
		prediction = np.real(np.sum(np.conj(self.feature)*X, axis=1))
		#prediction = np.real(np.sum(np.dot(np.conj(self.diff),self.Sigma_inv)*(X - self.avg), axis=1))
		return prediction
	def predict(self, X):
		prediction = self.dimreduce(X)>self.threshold
		return prediction
