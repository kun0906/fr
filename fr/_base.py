import copy
import time

import numpy as np
import scipy.stats
import sklearn
from scipy import linalg

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def trustworthiness(X, X_embedded, *, n_neighbors=5, metric="euclidean"):
	r"""Expresses to what extent the local structure is retained.

	The trustworthiness is within [0, 1]. It is defined as

	.. math::

		T(k) = 1 - \frac{2}{nk (2n - 3k - 1)} \sum^n_{i=1}
			\sum_{j \in \mathcal{N}_{i}^{k}} \max(0, (r(i, j) - k))

	where for each sample i, :math:`\mathcal{N}_{i}^{k}` are its k nearest
	neighbors in the output space, and every sample j is its :math:`r(i, j)`-th
	nearest neighbor in the input space. In other words, any unexpected nearest
	neighbors in the output space are penalised in proportion to their rank in
	the input space.

	Parameters
	----------
	X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
		If the metric is 'precomputed' X must be a square distance
		matrix. Otherwise it contains a sample per row.

	X_embedded : ndarray of shape (n_samples, n_components)
		Embedding of the training data in low-dimensional space.

	n_neighbors : int, default=5
		The number of neighbors that will be considered. Should be fewer than
		`n_samples / 2` to ensure the trustworthiness to lies within [0, 1], as
		mentioned in [1]_. An error will be raised otherwise.

	metric : str or callable, default='euclidean'
		Which metric to use for computing pairwise distances between samples
		from the original input space. If metric is 'precomputed', X must be a
		matrix of pairwise distances or squared distances. Otherwise, for a list
		of available metrics, see the documentation of argument metric in
		`sklearn.pairwise.pairwise_distances` and metrics listed in
		`sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`. Note that the
		"cosine" metric uses :func:`~sklearn.metrics.pairwise.cosine_distances`.

		.. versionadded:: 0.20

	Returns
	-------
	trustworthiness : float
		Trustworthiness of the low-dimensional embedding.

	References
	----------
	.. [1] Jarkko Venna and Samuel Kaski. 2001. Neighborhood
		   Preservation in Nonlinear Projection Methods: An Experimental Study.
		   In Proceedings of the International Conference on Artificial Neural Networks
		   (ICANN '01). Springer-Verlag, Berlin, Heidelberg, 485-491.

	.. [2] Laurens van der Maaten. Learning a Parametric Embedding by Preserving
		   Local Structure. Proceedings of the Twelth International Conference on
		   Artificial Intelligence and Statistics, PMLR 5:384-391, 2009.
	"""
	n_samples = X.shape[0]
	if n_neighbors >= n_samples / 2:
		raise ValueError(
			f"n_neighbors ({n_neighbors}) should be less than n_samples / 2"
			f" ({n_samples / 2})"
		)
	dist_X = pairwise_distances(X, metric=metric)
	if metric == "precomputed":
		dist_X = dist_X.copy()
	# we set the diagonal to np.inf to exclude the points themselves from
	# their own neighborhood
	np.fill_diagonal(dist_X, np.inf)
	ind_X = np.argsort(dist_X, axis=1)
	# `ind_X[i]` is the index of sorted distances between i and other samples
	ind_X_embedded = (
		NearestNeighbors(n_neighbors=n_neighbors)
			.fit(X_embedded)
			.kneighbors(return_distance=False)
	)

	# We build an inverted index of neighbors in the input space: For sample i,
	# we define `inverted_index[i]` as the inverted index of sorted distances:
	# inverted_index[i][ind_X[i]] = np.arange(1, n_sample + 1)
	inverted_index = np.zeros((n_samples, n_samples), dtype=int)
	ordered_indices = np.arange(n_samples + 1)
	inverted_index[ordered_indices[:-1, np.newaxis], ind_X] = ordered_indices[1:]
	ranks = (
			inverted_index[ordered_indices[:-1, np.newaxis], ind_X_embedded] - n_neighbors
	)
	t = np.sum(ranks[ranks > 0])
	t = 1.0 - t * (
			2.0 / (n_samples * n_neighbors * (2.0 * n_samples - 3.0 * n_neighbors - 1.0))
	)
	return t


def evaluate(X, y=None, X_embedded=None):
	res = {}

	# Get trustworthiness
	score = trustworthiness(X, X_embedded, n_neighbors=5, metric='euclidean')
	res['trustworthiness'] = score

	# Get continuity.
	score = trustworthiness(X_embedded, X, n_neighbors=5, metric='euclidean')
	res['continuity'] = score

	# Get 1-nearest neighbor accuracy
	nn = NearestNeighbors(n_neighbors=1)
	nn.fit(X_embedded)
	indices = nn.kneighbors(return_distance=False)  # exclude the query node itself
	y_pred = y[indices]
	acc = sklearn.metrics.accuracy_score(y, y_pred)
	res['acc_1nn'] = acc

	# Get Neighborhood hit based on K-nearest neighbor
	k = 5
	nn = NearestNeighbors(n_neighbors=k)
	nn.fit(X_embedded)
	indices = nn.kneighbors(return_distance=False)  # exclude the query node itself
	nh = 0
	y_preds = y[indices]
	n, d = X.shape
	for i in range(n):
		s = 0
		for l in y_preds[i]:
			if l == y[i]: s+=1
		nh += s/k
	res['neighborhood_hit'] = nh/n

	# Get normalized stress
	n, d = X.shape
	dist_X = pairwise_distances(X, X, metric="euclidean")
	dist_Y = pairwise_distances(X_embedded, X_embedded, metric="euclidean")
	ns = 0
	for i in range(n):
		for j in range(n):
			ns += (dist_X[i][j] - dist_Y[i][j])**2
	res['normalized_stress'] = ns / np.sum(np.square(dist_X))

	# shepard diagram goodness
	dist_X = pairwise_distances(X, X)
	n, _ = X.shape
	dist_X = dist_X[np.triu_indices(n, k=1)]   # upper triangle matrix without diagonal items
	dist_Y = pairwise_distances(X_embedded, X_embedded)
	dist_Y = dist_Y[np.triu_indices(n, k=1)]
	res['spearman'] = scipy.stats.spearmanr(dist_X, dist_Y)
	res['pearson'] = scipy.stats.pearsonr(dist_X, dist_Y)

	return res


def _gradient_descent(
		objective,
		p0, X, y,
		it,
		n_iter,
		n_iter_check=1,
		n_iter_without_progress=300,
		momentum=0.8,
		learning_rate=200.0,
		min_gain=0.01,
		min_grad_norm=1e-7,
		verbose=0,
		args=None,
		kwargs=None,
):
	"""Batch gradient descent with momentum and individual gains.

	Parameters
	----------
	objective : callable
		Should return a tuple of cost and gradient for a given parameter
		vector. When expensive to compute, the cost can optionally
		be None and can be computed every n_iter_check steps using
		the objective_error function.

	p0 : array-like of shape (n_params,)
		Initial parameter vector.

	it : int
		Current number of iterations (this function will be called more than
		once during the optimization).

	n_iter : int
		Maximum number of gradient descent iterations.

	n_iter_check : int, default=1
		Number of iterations before evaluating the global error. If the error
		is sufficiently low, we abort the optimization.

	n_iter_without_progress : int, default=300
		Maximum number of iterations without progress before we abort the
		optimization.

	momentum : float within (0.0, 1.0), default=0.8
		The momentum generates a weight for previous gradients that decays
		exponentially.

	learning_rate : float, default=200.0
		The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
		the learning rate is too high, the data may look like a 'ball' with any
		point approximately equidistant from its nearest neighbours. If the
		learning rate is too low, most points may look compressed in a dense
		cloud with few outliers.

	min_gain : float, default=0.01
		Minimum individual gain for each parameter.

	min_grad_norm : float, default=1e-7
		If the gradient norm is below this threshold, the optimization will
		be aborted.

	verbose : int, default=0
		Verbosity level.

	args : sequence, default=None
		Arguments to pass to objective function.

	kwargs : dict, default=None
		Keyword arguments to pass to objective function.

	Returns
	-------
	p : ndarray of shape (n_params,)
		Optimum parameters.

	error : float
		Optimum.

	i : int
		Last iteration.
	"""
	if args is None:
		args = []
	if kwargs is None:
		kwargs = {}

	p = p0.copy().ravel()
	update = np.zeros_like(p)
	gains = np.ones_like(p)
	error = np.finfo(float).max
	best_error = np.finfo(float).max
	best_iter = i = it
	res = []  # store the values of each iteration

	tic = time.time()
	for i in range(it, n_iter):
		i_start_time = time.time()
		check_convergence = (i + 1) % n_iter_check == 0
		# only compute the error when needed
		kwargs["compute_error"] = check_convergence or i == n_iter - 1

		error, grad = objective(p, *args, **kwargs)
		grad_norm = linalg.norm(grad)

		inc = update * grad < 0.0
		dec = np.invert(inc)
		gains[inc] += 0.2
		gains[dec] *= 0.8
		np.clip(gains, min_gain, np.inf, out=gains)
		grad *= gains
		update = momentum * update - learning_rate * grad
		p += update

		X_embedded = copy.deepcopy(p.reshape(X.shape[0], 2))
		# scores = evaluate(X, y, X_embedded)   # for debugging, it will cost too much time
		scores = {} # for saving time.
		i_end_time = time.time()
		res.append({'error': error, 'grad': grad, 'update': copy.deepcopy(update),
		            'Y': X_embedded, 'scores': scores, 'i_iter_duration': i_end_time-i_start_time})

		if check_convergence:
			toc = time.time()
			duration = toc - tic
			tic = toc

			if verbose >= 2:
				print(
					"[t-SNE] Iteration %d: error = %.7f,"
					" gradient norm = %.7f"
					" (%s iterations in %0.3fs)"
					% (i + 1, error, grad_norm, n_iter_check, duration)
				)

			if error < best_error:
				best_error = error
				best_iter = i
			elif i - best_iter > n_iter_without_progress:
				if verbose >= 2:
					print(
						"[t-SNE] Iteration %d: did not make any progress "
						"during the last %d episodes. Finished."
						% (i + 1, n_iter_without_progress)
					)
				break
			if grad_norm <= min_grad_norm:
				if verbose >= 2:
					print(
						"[t-SNE] Iteration %d: gradient norm %f. Finished."
						% (i + 1, grad_norm)
					)
				break

	return p, error, i, res

