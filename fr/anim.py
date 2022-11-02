import matplotlib.pyplot as plt
import moviepy.editor as mpy
# We'll generate an animation with matplotlib and moviepy.
import numpy as np
from moviepy.video.io.bindings import mplfig_to_npimage

X = []
y = []
for t in range(10):
	X_iter = np.asarray([1 * t, 1.3])
	y_iter = np.asarray([3, 4])
	X.append(X_iter)
	y.append(y_iter)

# here we are creating sub plots
fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(X[0], y[0])
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])

def make_frame_mpl(t):
	t = int(t)

	ax.set_title(f'{t}')
	sc.set_offsets(np.vstack([X[t], y[t]]))  # change position
	sc.set_sizes([20, 20])  # change sizes
	sc.set_array([0, 1])  # change color
	# https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot
	return mplfig_to_npimage(fig)


animation = mpy.VideoClip(make_frame_mpl, duration=10)
animation.write_gif("tmp.gif", fps=1)
