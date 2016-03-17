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
import annoy
import pickle

f = open('3d.pickle')
example1 = pickle.load(f)
f.close()

#example10 = []
#for i in range(example1.shape[2]):
#    example10.append(skimage.transform.rescale(example1, [32.0 / example1.shape[0], 32.0 / example1.shape[0], 32.0 / example1.shape[0]])

example10 = scipy.ndimage.zoom(example1, 20.0 / example1.shape[0])

example = example10[:, :64, :64]

for i in range(example.shape[0]):
    plt.imshow(example[i, :, :], interpolation = 'NONE', cmap = plt.cm.gray)
    plt.show()


p = 7

sampler = numpy.ones((p / 2 + 1, p, p))

vectors2 = annoy.AnnoyIndex(p * p * (p / 2 + 1))

sampler[p / 2, p / 2, p / 2:] = 0
sampler[p / 2, p / 2 + 1:, :] = 0

indices = []
vectors = []

for k in range(p / 2, example.shape[0] - p / 2):
    for i in range(p / 2, example.shape[1] - p / 2):
        for j in range(p / 2, example.shape[2] - p / 2):
            indices.append((k, i, j))

            vector = (sampler * example[k - p / 2 : k + 1, i - p / 2 : i + p / 2 + 1, j - p / 2 : j + p / 2 + 1]).flatten()

            vectors.append(vector)
            vectors2.add_item(len(indices) - 1, vector)
    print k

vectors2.build(5)

vectors = numpy.array(vectors)


test = numpy.random.randn(example.shape[0] + 4 * p, example.shape[1] + 4 * p, example.shape[2] + 4 * p) * numpy.std(example.flatten()) + 0.5 + numpy.mean(example.flatten())

#test = numpy.random.randn(32, 128, 128) * numpy.std(example.flatten()) + numpy.mean(example.flatten())

#numpy.array(exemplar)
vecnorms = numpy.linalg.norm(vectors, axis = 1)

idxes = []
for k in range(p / 2, test.shape[0] - p / 2):
    for i in range(p / 2, test.shape[1] - p / 2):
        for j in range(p / 2, test.shape[2] - p / 2):
            vec = (sampler * test[k - p / 2 : k + 1, i - p / 2 : i + p / 2 + 1, j - p / 2 : j + p / 2 + 1]).flatten()

            idx = vectors2.get_nns_by_vector(vec, 1)[0]

            test[k, i, j] = example[indices[idx]]
        #print 'Generating row: ', i
    print k

#plt.imshow(exemplar, cmap = plt.cm.gray, interpolation = 'NONE')
#plt.show()

test2 = scipy.ndimage.zoom(test[2 * p : -2 * p, 2 * p : -2 * p, 2 * p : -2 * p], test.shape[0] / 20.0)
test2 -= test2.flatten().min()
test2 /= test2.flatten().max()
test3 = scipy.ndimage.zoom(example, test.shape[0] / 20.0)
for k in range(test2.shape[0]):
    plt.imshow(test2[k, :, :], cmap = plt.cm.gray, interpolation = 'NONE')
    plt.show()
    skimage.io.imsave('3d/{0}.png'.format(k), test2[k, :, :])
    plt.imshow(test3[k, :, :], cmap = plt.cm.gray, interpolation = 'NONE')
    plt.show()
