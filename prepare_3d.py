#%%

import skimage.io
import skimage.transform
import os
import re
import numpy
import pickle

os.chdir('/home/bbales2/texture_generation')

#b = '/home/bbales2/virt/segmentation/ReneN4_FIB_Dataset/tosegment/cropped'
b = '/home/bbales2/virt/ch/output_1024/n_stack100'

images = []

for f in os.listdir(b):
    if re.search('.png', f):
        number = int(re.search('([0-9]+)', f).groups()[0])

        im = skimage.io.imread(os.path.join(b, f))

        im = skimage.transform.rescale(im, 1.0 / 2.255002151)#6.826666667

        images.append((number, im))

        print f

images = sorted(images, key = lambda x : x[0])

#%%

keys = [key for key, value in images]

stack = [value for key, value in images]

stack = numpy.array(stack)

f = open('3d.pickle', 'w')
pickle.dump(stack, f)
f.close()

#for ind, im in images:
#    skimage.io.imsave('3d/{0}.png'.format(ind), im)
#    print ind