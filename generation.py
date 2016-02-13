# This code is stolen from the algorithm description in here: https://graphics.stanford.edu/papers/texture-synthesis-sig00/texture.pdf
#
# It's quite a bit simplified
#
# The image is some Rene N4 from Will and McLean!
#%%

import skimage.io
import numpy
import scipy.ndimage
import os
import matplotlib.pyplot as plt

try:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exemplar5.png')
except:
    path = '/home/bbales2/texture_generation/exemplar5.png'

exemplar = skimage.io.imread(path, as_grey = True)

#%%

sampler = numpy.ones((7, 7))

sampler[4:, :] = 0
sampler[3, 3:] = 0

def sample(data):
    print data.shape

indices = []
vectors = []
for i in range(3, exemplar.shape[0] - 3):
    for j in range(3, exemplar.shape[1] - 3):
        indices.append((i, j))
        vectors.append((sampler * exemplar[i - 3 : i + 4, j - 3 : j + 4]).flatten())

vectors = numpy.array(vectors)
#%%
test = numpy.random.randn(100, 100) * numpy.std(exemplar.flatten()) + numpy.mean(exemplar.flatten())

#numpy.array(exemplar)
vecnorms = numpy.linalg.norm(vectors, axis = 1)

idxes = []
for i in range(3, test.shape[0] - 3):
    for j in range(3, test.shape[1] - 3):
        vec = (sampler * test[i - 3 : i + 4, j - 3 : j + 4]).flatten()

        idx = numpy.argmax(vectors.dot(vec) / (vecnorms * numpy.linalg.norm(vec)))

        test[i, j] = exemplar[indices[idx]]
    print 'Generating row: ', i
#%%
#plt.imshow(exemplar, cmap = plt.cm.gray, interpolation = 'NONE')
#plt.show()

plt.imshow(test[3 : test.shape[0] - 3, 3 : test.shape[1] - 3], cmap = plt.cm.gray, interpolation = 'NONE')
#fig = plt.gcf()
#fig.set_size_inches(18.5, 18.5)
plt.show()
