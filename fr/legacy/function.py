import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

for i in range(5):
	fig, ax = plt.subplots(2, 2, figsize=(10, 5), dpi=600)  # width, height
	print(f'i: {i}, figure number: {plt.gcf().number}, {fig.number}')
	# plt.close()

# https://stackoverflow.com/questions/52108558/how-does-parameters-c-and-cmap-behave-in-a-matplotlib-scatter-plot
N =100
x = [i for i in range(N)]
y = [1 for i in range(N)]
cmap = plt.get_cmap() # default colormap: viridis
norm = matplotlib.colors.Normalize(vmin=min(x), vmax=max(x))
colors = cmap(norm(x), bytes=False)   # RGBA
plt.scatter(x, y, c=colors)

y = [3 for i in range(N)]
cmap = plt.get_cmap(name='rainbow')
norm = matplotlib.colors.Normalize(vmin=min(x), vmax=max(x))
colors = cmap(norm(x), bytes=False)   # RGBA
plt.scatter(x, y, c=colors)

plt.show()

#
# N = 100
# r0 = 0.6
# x = 0.9 * np.random.rand(N)
# y = 0.9 * np.random.rand(N)
# area = (20 * np.random.rand(N))**2  # 0 to 10 point radii
# c = np.sqrt(area)
# r = np.sqrt(x ** 2 + y ** 2)
# area1 = np.ma.masked_where(r < r0, area)
# area2 = np.ma.masked_where(r >= r0, area)
# # plt.scatter(x, y, s=area1)
# cmap = plt.get_cmap(name='rainbow')
# norm = matplotlib.colors.Normalize(vmin=min(y), vmax=max(y))
# colors = cmap(norm(y), bytes=False)   # RGBA
# plt.scatter(x, y, s=area1, marker='^', c=colors)
# # plt.scatter(x, y, s=area2, marker='o', c=c)
# # Show the boundary between the regions:
# theta = np.arange(0, np.pi / 2, 0.01)
# plt.plot(r0 * np.cos(theta), r0 * np.sin(theta))
#
# plt.show()
#
#
#

# import matplotlib.pyplot as plt
# import numpy as np
#
# # def f3(v):
# x = [i/100 for i in range(1, 100)]
# # y = [np.exp(-v**2) for v in x]
# # plt.plot(x, y, 'r+', label='exp')
# # y = [1/(1+v) for v in x]
# # plt.plot(x, y, 'b+', label='2')
# # y = [(np.exp(v)-1)**(-1) for v in x]
# # plt.plot(x, y)
# # y = x
# # y = [(v**2)**(-1) for v in x]
# # plt.plot(x, y, label='v**2')
# # plt.plot(x, y, label='original')
# for base in [0.1, 0.5, 1, 1.5, 2, 3]:
# 	y = [base**v-1 for v in x]
# 	plt.plot(x, y,  label=base)
#
# #
# # for p in [1]: #[0.1, 0.5, 1, 2, 3, 4, 10]:
# # 	y = [1 / (1 + v**p) for v in x]
# # 	y = [np.log10(v) for v in x]
# # 	plt.plot(x, y, label=f'10')
# # 	y = [np.log2(v) for v in x]
# # 	plt.plot(x, y, label=f'2')
#
# plt.legend()
# plt.show()
#
# #
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
