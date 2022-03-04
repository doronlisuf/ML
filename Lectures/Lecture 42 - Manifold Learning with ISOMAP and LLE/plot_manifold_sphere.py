#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=============================================
Manifold Learning methods on a severed sphere
=============================================

An application of the different :ref:`manifold` techniques
on a spherical data-set. Here one can see the use of
dimensionality reduction in order to gain some intuition
regarding the manifold learning methods. Regarding the dataset,
the poles are cut from the sphere, as well as a thin slice down its
side. This enables the manifold learning techniques to
'spread it open' whilst projecting it onto two dimensions.

For a similar example, where the methods are applied to the
S-curve dataset, see :ref:`example_manifold_plot_compare_methods.py`

Note that the purpose of the :ref:`MDS <multidimensional_scaling>` is
to find a low-dimensional representation of the data (here 2D) in
which the distances respect well the distances in the original
high-dimensional space, unlike other manifold-learning algorithms,
it does not seeks an isotropic representation of the data in
the low-dimensional space. Here the manifold problem matches fairly
that of representing a flat map of the Earth, as with
`map projection <http://en.wikipedia.org/wiki/Map_projection>`_
"""

# Author: Jaques Grobler <jaques.grobler@inria.fr>
# License: BSD 3 clause

print(__doc__)

from time import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold
from sklearn.utils import check_random_state


# Next line to silence pyflakes.
Axes3D

# Variables for manifold learning.
n_neighbors = 8
n_samples = 1000

# Create our sphere.
random_state = check_random_state(0)
p = random_state.rand(n_samples) * (2 * np.pi - 0.55)
t = random_state.rand(n_samples) * np.pi

# Sever the poles from the sphere.
indices = ((t < (np.pi - (np.pi / 8))) & (t > ((np.pi / 8))))
colors = p[indices]
x, y, z = np.sin(t[indices]) * np.cos(p[indices]), \
    np.sin(t[indices]) * np.sin(p[indices]), \
    np.cos(t[indices])

# Plot our dataset.
fig = plt.figure(figsize=(15, 8))
# plt.suptitle("Manifold Learning with %i points, %i neighbors"
#              % (1000, n_neighbors), fontsize=14)

ax = fig.add_subplot(1,5,1, projection='3d')
ax.scatter(x, y, z, c=p[indices], cmap=plt.cm.rainbow)
try:
    # compatibility matplotlib < 1.0
    ax.view_init(40, -10)
except:
    pass

sphere_data = np.array([x, y, z]).T

# Perform PCA
t0 = time()
cov_mat = (sphere_data - np.mean(sphere_data,axis=0)).T.dot((sphere_data - np.mean(sphere_data,axis=0)))/np.size(sphere_data,0)
#Eigenvalues and Eigenvectors
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)
w_2D = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
y = sphere_data.dot(w_2D)
t1 = time()
print("%s: %.2g sec" % ('PCA', t1 - t0))

ax = fig.add_subplot(1,5,2)
plt.scatter(y[:,0], y[:,1], c=colors, cmap=plt.cm.rainbow)
plt.title("PCA (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

# Perform Multi-dimensional scaling.
t0 = time()
mds = manifold.MDS(2, max_iter=100, n_init=1)
trans_data = mds.fit_transform(sphere_data).T
t1 = time()
print("MDS: %.2g sec" % (t1 - t0))

ax = fig.add_subplot(1,5,3)
plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
# plt.title("MDS (%.2g sec)" % (t1 - t0))
plt.title('MDS',size=20)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

# Perform Isomap Manifold learning.
t0 = time()
trans_data = manifold.Isomap(n_neighbors, n_components=2)\
    .fit_transform(sphere_data).T
t1 = time()
print("%s: %.2g sec" % ('ISO', t1 - t0))

ax = fig.add_subplot(1,5,4)
plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
# plt.title("%s (%.2g sec)" % ('Isomap', t1 - t0))
plt.title('ISOMAP w/ '+str(n_neighbors)+' neighbors',size=20)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

# Perform LLE Manifold learning.
t0 = time()
trans_data = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2)\
    .fit_transform(sphere_data).T
t1 = time()
print("%s: %.2g sec" % ('ISO', t1 - t0))

ax = fig.add_subplot(1,5,5)
plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
# plt.title("%s (%.2g sec)" % ('LLE', t1 - t0))
plt.title('LLE w/ '+str(n_neighbors)+' neighbors',size=20)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

plt.show()