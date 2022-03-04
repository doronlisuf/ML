
#% Load Dependencies
import matplotlib.pyplot as plt
from matplotlib import offsetbox

import numpy as np

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.manifold import MDS

#%% Import Data
from keras.datasets import mnist

(images, labels), (images_test, labels_test) = mnist.load_data()

images = images/255.
images_test = images_test/255.

# Flattening image into vector
N,D,_ = images.shape
Ntest,D,_ = images_test.shape

X = images.flatten().reshape(N, D*D)
X_test = images_test.flatten().reshape(Ntest, D*D)

fig, ax = plt.subplots(6, 8, subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    axi.imshow(X[1250 * i].reshape(28, 28), cmap='gray_r')
    
#%%
# use only 1/40 of the data: full dataset takes a long time!

data = X[::40]
target = labels[::40]

#PCA
model = PCA(n_components=2)
proj = model.fit_transform(data)
fig, ax = plt.subplots(figsize=(10, 10))
plt.scatter(proj[:, 0], proj[:, 1], c=target, cmap=plt.cm.get_cmap('jet', 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5);
plt.title('PCA')

#LDA
model = LinearDiscriminantAnalysis(n_components=2)
proj = model.fit_transform(data, target)
fig, ax = plt.subplots(figsize=(10, 10))
plt.scatter(proj[:, 0], proj[:, 1], c=target, cmap=plt.cm.get_cmap('jet', 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5);
plt.title('LDA')

#MDS
model = MDS(n_components=2)
proj = model.fit_transform(data)
fig, ax = plt.subplots(figsize=(10, 10))
plt.scatter(proj[:, 0], proj[:, 1], c=target, cmap=plt.cm.get_cmap('jet', 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5);
plt.title('MDS')

# IsoMap
model = Isomap(n_components=2)
proj = model.fit_transform(data)
fig, ax = plt.subplots(figsize=(10, 10))
plt.scatter(proj[:, 0], proj[:, 1], c=target, cmap=plt.cm.get_cmap('jet', 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5);
plt.title('IsoMap')


#LLE
model = LLE(n_components=2)
proj = model.fit_transform(data)
fig, ax = plt.subplots(figsize=(10, 10))
plt.scatter(proj[:, 0], proj[:, 1], c=target, cmap=plt.cm.get_cmap('jet', 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5);
plt.title('LLE')

#%% Plotting scheme
# Source: https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html

def plot_components(data, model, images=None, ax=None,
                    thumb_frac=0.05, cmap='gray'):
    ax = ax or plt.gca()
    
    proj = model.fit_transform(data)
    ax.plot(proj[:, 0], proj[:, 1], '.k')
    
    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=cmap),
                                      proj[i])
            ax.add_artist(imagebox)

#%% 
# Choose 1/4 of the "1" digits to project
digit = 8

data = X[labels == digit][::4]

# PCA
# Choose 1/4 of the "1" digits to project
fig, ax = plt.subplots(figsize=(10, 10))
model = PCA(n_components=2)
plot_components(data, model, images=data.reshape((-1, 28, 28)),
                ax=ax, thumb_frac=0.05, cmap='gray_r')
plt.title('PCA')

# MDS
# Choose 1/4 of the "1" digits to project
fig, ax = plt.subplots(figsize=(10, 10))
model2 = MDS(n_components=2, max_iter=100, n_init=1)
plot_components(data, model2, images=data.reshape((-1, 28, 28)),
                ax=ax, thumb_frac=0.05, cmap='gray_r')
plt.title('MDS')

# ISOMAP
# Choose 1/4 of the "1" digits to project
fig, ax = plt.subplots(figsize=(10, 10))
model3 = Isomap(n_components=2)
plot_components(data, model3, images=data.reshape((-1, 28, 28)),
                ax=ax, thumb_frac=0.05, cmap='gray_r')
plt.title('ISOMAP')

# LLE
# Choose 1/4 of the "1" digits to project
fig, ax = plt.subplots(figsize=(10, 10))
model4 = LLE(n_components=2)
plot_components(data, model4, images=data.reshape((-1, 28, 28)),
                ax=ax, thumb_frac=0.05, cmap='gray_r')
plt.title('LLE')
