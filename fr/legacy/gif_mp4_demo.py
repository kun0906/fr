import os

import PIL
from PIL import Image
gif = []
out_dir = '../out/3gaussians-10dims/500/exact|30|weighted/(8, 22)|(4, 10)/batch'
figs = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if 'png' in f])
print(figs)
out_file  = 'temp_result.gif'
# images = [Image.open(f) for f in figs if 'png' in f] #images, just convert it into PIL.Image obj
# for image in images:
#     gif.append(image)
# gif[0].save(out_file, save_all=True,optimize=False, append_images=gif[1:], loop=2, duration=0.1)


print(figs)
import imageio
images = []
for i, f in enumerate(figs):
	im = imageio.v2.imread(f)   # RGBA
	if i == 0:
		shape = im.shape[:2][::-1]
	print(im.shape)
	# im.resize(shape)    # not work: https://stackoverflow.com/questions/65733362/how-to-resize-an-image-using-imageio
	im = PIL.Image.fromarray(im).resize(shape)  #  (width, height)
	images.append(im)
# kwargs = {'fps':1, 'loop':1}
kwargs = {'fps':1}
# imageio.mimsave(out_file, images, format='GIF', **kwargs)  # each image 0.5s = duration/n_imgs
# imageio.v2.mimsave(out_file, images, format='GIF', **kwargs)  # each image 0.5s = duration/n_imgs
imageio.v2.mimsave('a.mp4', images, format='mp4', **kwargs)  # each image 0.5s = duration/n_imgs


