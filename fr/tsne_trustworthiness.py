# Author: Alexander Fabisch  -- <afabisch@informatik.uni-bremen.de>
# Author: Christopher Moody <chrisemoody@gmail.com>
# Author: Nick Travers <nickt@squareup.com>
# License: BSD 3 clause (C) 2014

# This is the exact and Barnes-Hut t-SNE implementation. There are other
# modifications of the algorithm:
# * Fast Optimization for t-SNE:
#   https://cseweb.ucsd.edu/~lvdmaaten/workshops/nips2010/papers/vandermaaten.pdf
import copy
import time
import warnings

import numpy as np
from scipy import linalg
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.manifold._t_sne import _joint_probabilities, _joint_probabilities_nn, _kl_divergence_bh
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils.validation import check_non_negative, check_random_state

from _base import evaluate

MACHINE_EPSILON = np.finfo(np.double).eps


def dist2(x1, x2):
	return np.sum(np.square(x1, x2))


def _trustworthiness_loss_gradient(
		params,
		P,
		degrees_of_freedom,
		n_samples,
		n_components,
		skip_num_points=0,
		compute_error=True,
		X=None,
		knn_indices=None,
):
	"""t-SNE objective function: gradient of the KL divergence
	of p_ijs and q_ijs and the absolute error.

	Parameters
	----------
	params : ndarray of shape (n_params,)
		Unraveled embedding.

	P : ndarray of shape (n_samples * (n_samples-1) / 2,)
		Condensed joint probability matrix.

	degrees_of_freedom : int
		Degrees of freedom of the Student's-t distribution.

	n_samples : int
		Number of samples.

	n_components : int
		Dimension of the embedded space.

	skip_num_points : int, default=0
		This does not compute the gradient for points with indices below
		`skip_num_points`. This is useful when computing transforms of new
		data where you'd like to keep the old data fixed.

	compute_error: bool, default=True
		If False, the kl_divergence is not computed and returns NaN.

	Returns
	-------
	kl_divergence : float
		Kullback-Leibler divergence of p_ij and q_ij.

	grad : ndarray of shape (n_params,)
		Unraveled gradient of the Kullback-Leibler divergence with respect to
		the embedding.
	"""
	# A: a vector with shape (n_samples, )
	X_embedded, A = params
	# print(X_embedded.shape, A.shape)
	X_embedded = X_embedded.reshape(n_samples, n_components)
	# Objective: C
	if compute_error:
		loss = 0
		for i in range(n_samples):
			for j in knn_indices[i]:
				if i == j: continue # ignore the ponit itself
				d = dist2(X_embedded[i], X_embedded[j]) - A[i]
				if d > 0:
					loss += d
		loss = 0
	else:
		loss = np.nan

	# Gradient: dC/dY
	grad = np.ndarray((n_samples, n_components), dtype=X_embedded.dtype)
	for k in range(n_samples):
		g = 0
		# each data:
		y_k = X_embedded[k]
		# Gradient: dC/dp   for X_embedding.
		for i in range(n_samples):
			y_i = X_embedded[i]
			if i != k:
				if k in knn_indices[i] and dist2(y_i, y_k) > A[i]:  # if xk is the neighbor of xi
					g += 2*(y_i - y_k)
				else:
					pass
			else:  # i == k
				for j in knn_indices[i]:
					y_j = X_embedded[j]
					if dist2(y_i, y_j) > A[i]:
						g += 2*(y_i - y_j)
		grad[k] = g

		# Gradient: dC/dA
		grad_A = np.ndarray((n_samples,), dtype=A.dtype)
		for k in range(n_samples):
			g = 0
			# each data
			y_k = X_embedded[k]
			x_k = X[k]
			for i in range(n_samples):
				y_i = X_embedded[i]
				if i != k:
					if k in knn_indices[i] and dist2(y_i, y_k) > A[i]:  # if xk is the neighbor of xi
						g = g - 1
					else:
						pass
				else:  # i == k
					for j in knn_indices[i]:
						y_j = X_embedded[j]
						if dist2(y_i, y_j) > A[i]:
							g = g - 1
			grad_A[k] = g

	return loss, grad.ravel(), grad_A


def _gradient_descent(
		objective,
		p0, p1,
		X, y, knn_indices,
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

	kwargs['knn_indices'] = knn_indices
	kwargs['X'] = X

	p = [p0.copy(), p1.copy()]
	update = np.zeros_like(p[0])
	gains = np.ones_like(p[0])

	update_A = np.zeros_like(p[1])
	gains_A = np.ones_like(p[1])

	error = np.finfo(float).max
	best_error = np.finfo(float).max
	best_iter = i = it
	res = []  # store the values of each iteration

	tic = time.time()
	for i in range(it, n_iter):
		check_convergence = (i + 1) % n_iter_check == 0
		# only compute the error when needed
		kwargs["compute_error"] = check_convergence or i == n_iter - 1

		error, grad, grad_A = objective(p, *args, **kwargs)
		grad_norm = linalg.norm(grad)
		grad_A_norm = linalg.norm(grad_A)

		# https://github.com/scikit-learn/scikit-learn/pull/8768
		inc = update * grad < 0.0
		dec = np.invert(inc)
		gains[inc] += 0.2
		gains[dec] *= 0.8
		np.clip(gains, min_gain, np.inf, out=gains)
		grad *= gains
		update = momentum * update - learning_rate * grad
		p[0] += update

		inc = update_A * grad_A < 0.0
		dec = np.invert(inc)
		gains_A[inc] += 0.2
		gains_A[dec] *= 0.8
		np.clip(gains_A, min_gain, np.inf, out=gains_A)
		grad_A *= gains_A
		update_A = momentum * update_A - learning_rate * grad_A
		p[1] += update_A

		X_embedded = copy.deepcopy(p[0].reshape(X.shape[0], 2))
		scores = evaluate(X, y, X_embedded)
		res.append({'error': error, 'grad': grad, 'update': copy.deepcopy(update), 'Y': X_embedded, 'scores': scores})
		print(p[0].shape, p[1].shape, flush=True)
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

	return p[0], error, i, res


class TSNE(BaseEstimator):
	"""T-distributed Stochastic Neighbor Embedding.

	t-SNE [1] is a tool to visualize high-dimensional data. It converts
	similarities between data points to joint probabilities and tries
	to minimize the Kullback-Leibler divergence between the joint
	probabilities of the low-dimensional embedding and the
	high-dimensional data. t-SNE has a cost function that is not convex,
	i.e. with different initializations we can get different results.

	It is highly recommended to use another dimensionality reduction
	method (e.g. PCA for dense data or TruncatedSVD for sparse data)
	to reduce the number of dimensions to a reasonable amount (e.g. 50)
	if the number of features is very high. This will suppress some
	noise and speed up the computation of pairwise distances between
	samples. For more tips see Laurens van der Maaten's FAQ [2].

	Read more in the :ref:`User Guide <t_sne>`.

	Parameters
	----------
	n_components : int, default=2
		Dimension of the embedded space.

	perplexity : float, default=30.0
		The perplexity is related to the number of nearest neighbors that
		is used in other manifold learning algorithms. Larger datasets
		usually require a larger perplexity. Consider selecting a value
		between 5 and 50. Different values can result in significantly
		different results. The perplexity must be less that the number
		of samples.

	early_exaggeration : float, default=12.0
		Controls how tight natural clusters in the original space are in
		the embedded space and how much space will be between them. For
		larger values, the space between natural clusters will be larger
		in the embedded space. Again, the choice of this parameter is not
		very critical. If the cost function increases during initial
		optimization, the early exaggeration factor or the learning rate
		might be too high.

	learning_rate : float or 'auto', default=200.0
		The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
		the learning rate is too high, the data may look like a 'ball' with any
		point approximately equidistant from its nearest neighbours. If the
		learning rate is too low, most points may look compressed in a dense
		cloud with few outliers. If the cost function gets stuck in a bad local
		minimum increasing the learning rate may help.
		Note that many other t-SNE implementations (bhtsne, FIt-SNE, openTSNE,
		etc.) use a definition of learning_rate that is 4 times smaller than
		ours. So our learning_rate=200 corresponds to learning_rate=800 in
		those other implementations. The 'auto' option sets the learning_rate
		to `max(N / early_exaggeration / 4, 50)` where N is the sample size,
		following [4] and [5]. This will become default in 1.2.

	n_iter : int, default=1000
		Maximum number of iterations for the optimization. Should be at
		least 250.

	n_iter_without_progress : int, default=300
		Maximum number of iterations without progress before we abort the
		optimization, used after 250 initial iterations with early
		exaggeration. Note that progress is only checked every 50 iterations so
		this value is rounded to the next multiple of 50.

		.. versionadded:: 0.17
		   parameter *n_iter_without_progress* to control stopping criteria.

	min_grad_norm : float, default=1e-7
		If the gradient norm is below this threshold, the optimization will
		be stopped.

	metric : str or callable, default='euclidean'
		The metric to use when calculating distance between instances in a
		feature array. If metric is a string, it must be one of the options
		allowed by scipy.spatial.distance.pdist for its metric parameter, or
		a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
		If metric is "precomputed", X is assumed to be a distance matrix.
		Alternatively, if metric is a callable function, it is called on each
		pair of instances (rows) and the resulting value recorded. The callable
		should take two arrays from X as input and return a value indicating
		the distance between them. The default is "euclidean" which is
		interpreted as squared euclidean distance.

	metric_params : dict, default=None
		Additional keyword arguments for the metric function.

		.. versionadded:: 1.1

	init : {'random', 'pca'} or ndarray of shape (n_samples, n_components), \
			default='random'
		Initialization of embedding. Possible options are 'random', 'pca',
		and a numpy array of shape (n_samples, n_components).
		PCA initialization cannot be used with precomputed distances and is
		usually more globally stable than random initialization. `init='pca'`
		will become default in 1.2.

	verbose : int, default=0
		Verbosity level.

	random_state : int, RandomState instance or None, default=None
		Determines the random number generator. Pass an int for reproducible
		results across multiple function calls. Note that different
		initializations might result in different local minima of the cost
		function. See :term:`Glossary <random_state>`.

	method : str, default='barnes_hut'
		By default the gradient calculation algorithm uses Barnes-Hut
		approximation running in O(NlogN) time. method='exact'
		will run on the slower, but exact, algorithm in O(N^2) time. The
		exact algorithm should be used when nearest-neighbor errors need
		to be better than 3%. However, the exact method cannot scale to
		millions of examples.

		.. versionadded:: 0.17
		   Approximate optimization *method* via the Barnes-Hut.

	angle : float, default=0.5
		Only used if method='barnes_hut'
		This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
		'angle' is the angular size (referred to as theta in [3]) of a distant
		node as measured from a point. If this size is below 'angle' then it is
		used as a summary node of all points contained within it.
		This method is not very sensitive to changes in this parameter
		in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
		computation time and angle greater 0.8 has quickly increasing error.

	n_jobs : int, default=None
		The number of parallel jobs to run for neighbors search. This parameter
		has no impact when ``metric="precomputed"`` or
		(``metric="euclidean"`` and ``method="exact"``).
		``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
		``-1`` means using all processors. See :term:`Glossary <n_jobs>`
		for more details.

		.. versionadded:: 0.22

	square_distances : True, default='deprecated'
		This parameter has no effect since distance values are always squared
		since 1.1.

		.. deprecated:: 1.1
			 `square_distances` has no effect from 1.1 and will be removed in
			 1.3.

	Attributes
	----------
	embedding_ : array-like of shape (n_samples, n_components)
		Stores the embedding vectors.

	kl_divergence_ : float
		Kullback-Leibler divergence after optimization.

	n_features_in_ : int
		Number of features seen during :term:`fit`.

		.. versionadded:: 0.24

	feature_names_in_ : ndarray of shape (`n_features_in_`,)
		Names of features seen during :term:`fit`. Defined only when `X`
		has feature names that are all strings.

		.. versionadded:: 1.0

	n_iter_ : int
		Number of iterations run.

	See Also
	--------
	sklearn.decomposition.PCA : Principal component analysis that is a linear
		dimensionality reduction method.
	sklearn.decomposition.KernelPCA : Non-linear dimensionality reduction using
		kernels and PCA.
	MDS : Manifold learning using multidimensional scaling.
	Isomap : Manifold learning based on Isometric Mapping.
	LocallyLinearEmbedding : Manifold learning using Locally Linear Embedding.
	SpectralEmbedding : Spectral embedding for non-linear dimensionality.

	References
	----------

	[1] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
		Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.

	[2] van der Maaten, L.J.P. t-Distributed Stochastic Neighbor Embedding
		https://lvdmaaten.github.io/tsne/

	[3] L.J.P. van der Maaten. Accelerating t-SNE using Tree-Based Algorithms.
		Journal of Machine Learning Research 15(Oct):3221-3245, 2014.
		https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf

	[4] Belkina, A. C., Ciccolella, C. O., Anno, R., Halpert, R., Spidlen, J.,
		& Snyder-Cappione, J. E. (2019). Automated optimized parameters for
		T-distributed stochastic neighbor embedding improve visualization
		and analysis of large datasets. Nature Communications, 10(1), 1-12.

	[5] Kobak, D., & Berens, P. (2019). The art of using t-SNE for single-cell
		transcriptomics. Nature Communications, 10(1), 1-14.

	Examples
	--------
	>>> import numpy as np
	>>> from sklearn.manifold import TSNE
	>>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
	>>> X_embedded = TSNE(n_components=2, learning_rate='auto',
	...                   init='random', perplexity=3).fit_transform(X)
	>>> X_embedded.shape
	(4, 2)
	"""

	# Control the number of exploration iterations with early_exaggeration on
	# _EXPLORATION_N_ITER = 250

	# Control the number of iterations between progress checks
	_N_ITER_CHECK = 50

	def __init__(
			self,
			n_components=2,
			*,
			perplexity=30.0,
			early_exaggeration=12.0,
			learning_rate="warn",
			n_iter=1000,
			n_iter_without_progress=300,
			min_grad_norm=1e-7,
			metric="euclidean",
			metric_params=None,
			init="warn",
			verbose=0,
			random_state=None,
			method="barnes_hut",
			angle=0.5,
			n_jobs=None,
			square_distances="deprecated",
			_EXPLORATION_N_ITER=250,
	):
		self.n_components = n_components
		self.perplexity = perplexity
		self.early_exaggeration = early_exaggeration
		self.learning_rate = learning_rate
		self.n_iter = n_iter
		self.n_iter_without_progress = n_iter_without_progress
		self.min_grad_norm = min_grad_norm
		self.metric = metric
		self.metric_params = metric_params
		self.init = init
		self.verbose = verbose
		self.random_state = random_state
		self.method = method
		self.angle = angle
		self.n_jobs = n_jobs
		self.square_distances = square_distances
		self._EXPLORATION_N_ITER = _EXPLORATION_N_ITER

	def _check_params_vs_input(self, X):
		if self.perplexity >= X.shape[0]:
			raise ValueError("perplexity must be less than n_samples")

	def _fit(self, X, y=None, skip_num_points=0):
		"""Private function to fit the model using X as training data."""

		if isinstance(self.init, str) and self.init == "warn":
			# See issue #18018
			warnings.warn(
				"The default initialization in TSNE will change "
				"from 'random' to 'pca' in 1.2.",
				FutureWarning,
			)
			self._init = "random"
		else:
			self._init = self.init
		if self.learning_rate == "warn":
			# See issue #18018
			warnings.warn(
				"The default learning rate in TSNE will change "
				"from 200.0 to 'auto' in 1.2.",
				FutureWarning,
			)
			self._learning_rate = 200.0
		else:
			self._learning_rate = self.learning_rate

		if isinstance(self._init, str) and self._init == "pca" and issparse(X):
			raise TypeError(
				"PCA initialization is currently not supported "
				"with the sparse input matrix. Use "
				'init="random" instead.'
			)
		if self.method not in ["barnes_hut", "exact", "trustworthiness"]:
			raise ValueError("'method' must be 'barnes_hut' or 'exact'")
		if self.angle < 0.0 or self.angle > 1.0:
			raise ValueError("'angle' must be between 0.0 - 1.0")
		if self.square_distances != "deprecated":
			warnings.warn(
				"The parameter `square_distances` has not effect and will be "
				"removed in version 1.3.",
				FutureWarning,
			)
		if self._learning_rate == "auto":
			# See issue #18018
			self._learning_rate = X.shape[0] / self.early_exaggeration / 4
			self._learning_rate = np.maximum(self._learning_rate, 50)
		else:
			if not (self._learning_rate > 0):
				raise ValueError("'learning_rate' must be a positive number or 'auto'.")
		if self.method == "barnes_hut":
			X = self._validate_data(
				X,
				accept_sparse=["csr"],
				ensure_min_samples=2,
				dtype=[np.float32, np.float64],
			)
		else:
			X = self._validate_data(
				X, accept_sparse=["csr", "csc", "coo"], dtype=[np.float32, np.float64]
			)
		if self.metric == "precomputed":
			if isinstance(self._init, str) and self._init == "pca":
				raise ValueError(
					'The parameter init="pca" cannot be used with metric="precomputed".'
				)
			if X.shape[0] != X.shape[1]:
				raise ValueError("X should be a square distance matrix")

			check_non_negative(
				X,
				"TSNE.fit(). With metric='precomputed', X "
				"should contain positive distances.",
			)

			if self.method == "exact" and issparse(X):
				raise TypeError(
					'TSNE with method="exact" does not accept sparse '
					'precomputed distance matrix. Use method="barnes_hut" '
					"or provide the dense distance matrix."
				)

		if self.method == "barnes_hut" and self.n_components > 3:
			raise ValueError(
				"'n_components' should be inferior to 4 for the "
				"barnes_hut algorithm as it relies on "
				"quad-tree or oct-tree."
			)
		random_state = check_random_state(self.random_state)

		if self.early_exaggeration < 1.0:
			raise ValueError(
				"early_exaggeration must be at least 1, but is {}".format(
					self.early_exaggeration
				)
			)

		# if self.n_iter < 250:
		#     raise ValueError("n_iter should be at least 250")

		n_samples = X.shape[0]

		neighbors_nn = None
		if self.method == "exact":
			st = time.time()
			# Retrieve the distance matrix, either using the precomputed one or
			# computing it.
			if self.metric == "precomputed":
				distances = X
			else:
				if self.verbose:
					print("[t-SNE] Computing pairwise distances...")

				if self.metric == "euclidean":
					# Euclidean is squared here, rather than using **= 2,
					# because euclidean_distances already calculates
					# squared distances, and returns np.sqrt(dist) for
					# squared=False.
					# Also, Euclidean is slower for n_jobs>1, so don't set here
					distances = pairwise_distances(X, metric=self.metric, squared=True)
				else:
					metric_params_ = self.metric_params or {}
					distances = pairwise_distances(
						X, metric=self.metric, n_jobs=self.n_jobs, **metric_params_
					)

			if np.any(distances < 0):
				raise ValueError(
					"All distances should be positive, the metric given is not correct"
				)

			if self.metric != "euclidean":
				distances **= 2
			ed = time.time()
			update_dist_time = ed - st

			st = time.time()
			# compute the joint probability distribution for the input space
			P = _joint_probabilities(distances, self.perplexity, self.verbose)
			assert np.all(np.isfinite(P)), "All probabilities should be finite"
			assert np.all(P >= 0), "All probabilities should be non-negative"
			assert np.all(
				P <= 1
			), "All probabilities should be less or then equal to one"
			ed = time.time()
			update_P_time = ed - st
		elif self.method == 'trustworthiness':
			# Compute the number of nearest neighbors to find.
			# LvdM uses 3 * perplexity as the number of neighbors.
			# In the event that we have very small # of points
			# set the neighbors to n - 1.
			n_neighbors = 1 + 5  # i.e., k = 1+5 including the point itself.

			if self.verbose:
				print("[t-SNE] Computing {} nearest neighbors...".format(n_neighbors))

			# Find the nearest neighbors for every point
			knn = NearestNeighbors(
				algorithm="auto",
				n_jobs=self.n_jobs,
				n_neighbors=n_neighbors,
				metric=self.metric,
				metric_params=self.metric_params,
			)
			t0 = time.time()
			knn.fit(X)
			duration = time.time() - t0
			if self.verbose:
				print(
					"[t-SNE] Indexed {} samples in {:.3f}s...".format(
						n_samples, duration
					)
				)
			P = 0
			update_dist_time = 0
			update_P_time = 0
		else:
			# Compute the number of nearest neighbors to find.
			# LvdM uses 3 * perplexity as the number of neighbors.
			# In the event that we have very small # of points
			# set the neighbors to n - 1.
			n_neighbors = min(n_samples - 1, int(3.0 * self.perplexity + 1))

			if self.verbose:
				print("[t-SNE] Computing {} nearest neighbors...".format(n_neighbors))

			# Find the nearest neighbors for every point
			knn = NearestNeighbors(
				algorithm="auto",
				n_jobs=self.n_jobs,
				n_neighbors=n_neighbors,
				metric=self.metric,
				metric_params=self.metric_params,
			)
			t0 = time.time()
			knn.fit(X)
			duration = time.time() - t0
			if self.verbose:
				print(
					"[t-SNE] Indexed {} samples in {:.3f}s...".format(
						n_samples, duration
					)
				)

			t0 = time.time()
			distances_nn = knn.kneighbors_graph(mode="distance")
			duration = time.time() - t0
			if self.verbose:
				print(
					"[t-SNE] Computed neighbors for {} samples in {:.3f}s...".format(
						n_samples, duration
					)
				)

			# Free the memory used by the ball_tree
			del knn

			# knn return the euclidean distance but we need it squared
			# to be consistent with the 'exact' method. Note that the
			# the method was derived using the euclidean method as in the
			# input space. Not sure of the implication of using a different
			# metric.
			distances_nn.data **= 2

			# compute the joint probability distribution for the input space
			P = _joint_probabilities_nn(distances_nn, self.perplexity, self.verbose)

		st = time.time()
		if isinstance(self._init, np.ndarray):
			X_embedded = self._init
		elif self._init == "pca":
			pca = PCA(
				n_components=self.n_components,
				svd_solver="randomized",
				random_state=random_state,
			)
			X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
			# TODO: Update in 1.2
			# PCA is rescaled so that PC1 has standard deviation 1e-4 which is
			# the default value for random initialization. See issue #18018.
			warnings.warn(
				"The PCA initialization in TSNE will change to "
				"have the standard deviation of PC1 equal to 1e-4 "
				"in 1.2. This will ensure better convergence.",
				FutureWarning,
			)
		# X_embedded = X_embedded / np.std(X_embedded[:, 0]) * 1e-4
		elif self._init == "random":
			# The embedding is initialized with iid samples from Gaussians with
			# standard deviation 1e-4.
			X_embedded = 1e-4 * random_state.standard_normal(
				size=(n_samples, self.n_components)
			).astype(np.float32)
			A = 1e-4 * random_state.standard_normal(
				size=(n_samples,)
			).astype(np.float32)
		else:
			raise ValueError("'init' must be 'pca', 'random', or a numpy array")
		ed = time.time()
		concat_Y_time = ed - st

		st = time.time()
		# Degrees of freedom of the Student's t-distribution. The suggestion
		# degrees_of_freedom = n_components - 1 comes from
		# "Learning a Parametric Embedding by Preserving Local Structure"
		# Laurens van der Maaten, 2009.
		degrees_of_freedom = max(self.n_components - 1, 1)
		knn_dist, knn_indices = knn.kneighbors(X, n_neighbors=n_neighbors, return_distance=True)
		X_embedded, A = self._tsne(
			P,
			degrees_of_freedom,
			n_samples,
			X_embedded=X_embedded,
			A=A,
			neighbors=neighbors_nn,
			skip_num_points=skip_num_points,
			X=X,
			y=y,
			knn_indices=knn_indices,
		)
		ed = time.time()
		update_Y_time = ed - st
		self.fit_res = {'time': (update_dist_time, update_P_time, concat_Y_time, update_Y_time),
		                'kl_divergence': self.kl_divergence_lst}
		self.P = P
		return X_embedded

	def _tsne(
			self,
			P, # P distribution matrix
			degrees_of_freedom,
			n_samples,
			X_embedded, A,
			neighbors=None,
			skip_num_points=0,
			X=None,
			y=None,
			knn_indices=None,
	):
		"""Runs t-SNE."""
		# t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
		# and the Student's t-distributions Q. The optimization algorithm that
		# we use is batch gradient descent with two stages:
		# * initial optimization with early exaggeration and momentum at 0.5
		# * final optimization with momentum at 0.8
		params = X_embedded.ravel()

		opt_args = {
			"it": 0,
			"n_iter_check": self._N_ITER_CHECK,
			"min_grad_norm": self.min_grad_norm,
			"learning_rate": self._learning_rate,
			"verbose": self.verbose,
			"kwargs": dict(skip_num_points=skip_num_points),
			"args": [P, degrees_of_freedom, n_samples, self.n_components],
			"n_iter_without_progress": self._EXPLORATION_N_ITER,
			"n_iter": self._EXPLORATION_N_ITER,
			"momentum": 0.5,
		}
		if self.method == "barnes_hut":
			obj_func = _kl_divergence_bh
			opt_args["kwargs"]["angle"] = self.angle
			# Repeat verbose argument for _kl_divergence_bh
			opt_args["kwargs"]["verbose"] = self.verbose
			# Get the number of threads for gradient computation here to
			# avoid recomputing it at each iteration.
			opt_args["kwargs"]["num_threads"] = _openmp_effective_n_threads()
		elif self.method == 'trustworthiness':
			obj_func = _trustworthiness_loss_gradient
		else:
			raise NotImplementedError(f'{self.method} is not implemented.')

		self.kl_divergence_lst = []
		# Learning schedule (part 1): do 250 iteration with lower momentum but
		# higher learning rate controlled via the early exaggeration parameter
		P *= self.early_exaggeration
		st = time.time()
		params, kl_divergence, it, res1 = _gradient_descent(obj_func, params, A, X, y, knn_indices, **opt_args)
		ed = time.time()
		dur = ed - st
		pre_iter = it + 1
		self.kl_divergence_lst.append((kl_divergence, pre_iter, dur, dur / pre_iter))
		if self.verbose:
			print(
				"[t-SNE] KL divergence after %d iterations with early exaggeration: %f (%fs)"
				% (it + 1, kl_divergence, ed - st)
			)
		# Learning schedule (part 2): disable early exaggeration and finish
		# optimization with a higher momentum at 0.8
		P /= self.early_exaggeration
		st = time.time()
		remaining = self.n_iter - self._EXPLORATION_N_ITER
		if it < self._EXPLORATION_N_ITER or remaining > 0:
			opt_args["n_iter"] = self.n_iter
			opt_args["it"] = it + 1
			opt_args["momentum"] = 0.8
			opt_args["n_iter_without_progress"] = self.n_iter_without_progress
			params, kl_divergence, it, res2 = _gradient_descent(obj_func, params, A, X, y, knn_indices, **opt_args)

		# Save the final number of iterations
		ed = time.time()
		self.n_iter_ = it
		pre_iter = it - pre_iter + 1
		dur = ed - st

		self.kl_divergence_lst.append((kl_divergence, pre_iter, dur, dur / pre_iter))
		if self.verbose:
			print(
				"[t-SNE] KL divergence after %d iterations: %f (%fs)"
				% (it + 1, kl_divergence, ed - st)
			)

		X_embedded = params.reshape(n_samples, self.n_components)
		self.kl_divergence_ = kl_divergence
		self._update_data = (res1, res2)  # two updates' results
		return X_embedded, A

	def fit_transform(self, X, y=None):
		"""Fit X into an embedded space and return that transformed output.

		Parameters
		----------
		X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
			If the metric is 'precomputed' X must be a square distance
			matrix. Otherwise it contains a sample per row. If the method
			is 'exact', X may be a sparse matrix of type 'csr', 'csc'
			or 'coo'. If the method is 'barnes_hut' and the metric is
			'precomputed', X may be a precomputed sparse graph.

		y : None
			Ignored.

		Returns
		-------
		X_new : ndarray of shape (n_samples, n_components)
			Embedding of the training data in low-dimensional space.
		"""
		self._check_params_vs_input(X)
		embedding = self._fit(X, y)
		self.embedding_ = embedding
		return self.embedding_

	def fit(self, X, y=None):
		"""Fit X into an embedded space.

		Parameters
		----------
		X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
			If the metric is 'precomputed' X must be a square distance
			matrix. Otherwise it contains a sample per row. If the method
			is 'exact', X may be a sparse matrix of type 'csr', 'csc'
			or 'coo'. If the method is 'barnes_hut' and the metric is
			'precomputed', X may be a precomputed sparse graph.

		y : None
			Ignored.

		Returns
		-------
		X_new : array of shape (n_samples, n_components)
			Embedding of the training data in low-dimensional space.
		"""
		self.fit_transform(X, y)
		return self

	def _more_tags(self):
		return {"pairwise": self.metric == "precomputed"}
