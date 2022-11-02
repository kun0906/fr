
import matplotlib.pyplot as plt
import numpy as np

# def f3(v):
x = [i/10 for i in range(100)]
y = [np.exp(-v**2) for v in x]
plt.plot(x, y, 'r+', label='exp')
# y = [1/(1+v) for v in x]
# plt.plot(x, y, 'b+', label='2')


for p in [0.1, 0.5, 1, 2, 3, 4, 10]:
	y = [1 / (1 + v**p) for v in x]
	plt.plot(x, y, label=f'{p}')

plt.legend()
plt.show()

#
# def f1(base):
# 	a  = 10
# 	x = [i/100 for i in range(100)]
# 	# y = [v/np.sqrt(1+v**3) for v in x]
# 	# y = [1 / (1+np.exp(-v)) for v in x]
# 	# y = [np.exp(-v*base) for v in x]
# 	# y = [base**v-1 for v in x]
# 	# y = [v**base for v in x]
# 	# y = [(np.abs(v-0.5)*1)**base for v in x]
# 	y = [1/(v+1e-1)**base for v in x]
# 	# y = [(np.exp(v) - np.exp(-v))*base/(np.exp(v) + np.exp(-v)) for v in x]
# 	return x,y
#
# def f2():
# 	x = [i / 100 for i in range(100)]
# 	y = [np.exp(v)-1 for v in x]
#
# 	return x, y
#
# for base in [0.5, 0.1, 1, 0.9]:# [0.1, 0.5, 0.9, 1, 2,3, 5, 7,]: # 0.1, 0.3, 0.5, 0.9,
# 	x, y = f1(base)
# 	plt.plot(x, y,  label = f'{base}')
#
# x, y = f2()
# plt.plot(x, y, 'r+')
# plt.legend()
# plt.show()
