import numpy as np
import pandas
from skimage import img_as_float
from sklearn.cluster import KMeans
from skimage.io import imread, imsave

image = imread('samples/parrots.jpg')
norm_image = img_as_float(image)

height, width, deep = norm_image.shape

pixels = pandas.DataFrame(np.asarray(norm_image).reshape(height * width, deep), columns=['R', 'G', 'B'])

k_means = KMeans(init='k-means++', random_state=241)
pixels['cluster'] = result = k_means.fit_predict(pixels)


def apply_clustering(cluster_values):
    new_pixels = [cluster_values[v] for v in pixels['cluster'].values]
    new_image = np.reshape(new_pixels, (height, width, deep))
    return new_image


means = pixels.groupby('cluster').mean().values
mean_image = apply_clustering(means)
imsave('images/parrot_mean.png', mean_image)

medians = pixels.groupby('cluster').median().values
median_image = apply_clustering(medians)
imsave('images/parrot_median.png', median_image)
