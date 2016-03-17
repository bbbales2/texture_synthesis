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
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exemplar2.png')
except:
    path = '/home/bbales2/texture_generation/exemplar2.png'

exemplar = skimage.io.imread(path, as_grey = True)

#%%

p = 5

sampler = numpy.ones((p, p))

sampler[p / 2 + 1:, :] = 0
sampler[p / 2, p / 2:] = 0

def sample(data):
    print data.shape

indices = []
vectors = []

for i in range(p / 2 + 1, exemplar.shape[0] - p / 2):
    for j in range(p / 2 + 1, exemplar.shape[1] - p / 2):
        indices.append((i, j))
        vectors.append((sampler * exemplar[i - p / 2 : i + p / 2 + 1, j - p / 2 : j + p / 2 + 1]).flatten())

vectors = numpy.array(vectors)

test = numpy.random.randn(100, 100) * numpy.std(exemplar.flatten()) + numpy.mean(exemplar.flatten())

#numpy.array(exemplar)
vecnorms = numpy.linalg.norm(vectors, axis = 1)

idxes = []
for i in range(p / 2 + 1, test.shape[0] - p / 2):
    for j in range(p / 2 + 1, test.shape[1] - p / 2):
        vec = (sampler * test[i - p / 2 : i + p / 2 + 1, j - p / 2 : j + p / 2 + 1]).flatten()

        idx = numpy.argmax(vectors.dot(vec) / (vecnorms * numpy.linalg.norm(vec)))

        test[i, j] = exemplar[indices[idx]]
    print 'Generating row: ', i

#plt.imshow(exemplar, cmap = plt.cm.gray, interpolation = 'NONE')
#plt.show()

plt.imshow(test[p / 2 + 1 : test.shape[0] - p / 2, p / 2 + 1 : test.shape[1] - p / 2], cmap = plt.cm.gray, interpolation = 'NONE')
#fig = plt.gcf()
#fig.set_size_inches(18.5, 18.5)
plt.show()
