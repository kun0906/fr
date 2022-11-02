import numpy as np
import matplotlib.pyplot as plt


random_state = 42
is_show = True

rng = np.random.RandomState(seed=random_state)
cov1 = [[0.1, 0.0], [0.0, 0.1]]
cov2 = [[0., 0.1], [0.1, 0.]]
cov3 = [[0.1, 0.1], [0.1, 0.1]]
# cov3 = [[1, 1], [1, 1]]
cov4 = [[0.1, 0.], [0., 0.]]
# ============
# Set up cluster parameters
# ============
plt.figure()
# plt.subplots_adjust(
# 	left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
# )

for cov in [cov1, cov2, cov3, cov4]:

	X = rng.multivariate_normal(mean=[2, 0], cov=np.asarray(cov),
	                                  size=1000)
	y = np.asarray([2] * X.shape[0])
	plt.scatter(X[:, 0], X[:, 1], c=y)

	plt.title(cov)
	if is_show:
		plt.show()
plt.close()
print('test')
