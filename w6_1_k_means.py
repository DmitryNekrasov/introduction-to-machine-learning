import numpy as np
import pandas
from math import log10
from skimage import img_as_float
from sklearn.cluster import KMeans
from skimage.io import imread, imsave

image = imread('samples/parrots.jpg')
norm_image = img_as_float(image)

height, width, deep = norm_image.shape

pixels = pandas.DataFrame(np.asarray(norm_image).reshape(height * width, deep), columns=['R', 'G', 'B'])
pixels['cluster'] = 0


def apply_clustering(cluster_values):
    new_pixels = [cluster_values[v] for v in pixels['cluster'].values]
    new_image = np.reshape(new_pixels, (height, width, deep))
    return new_image


def get_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    return 10 * log10(1.0 / mse)


ans = -1
for nc in range(1, 21):
    k_means = KMeans(init='k-means++', random_state=241, n_clusters=nc)
    pixels['cluster'] = k_means.fit_predict(pixels.drop(['cluster'], axis=1))

    means = pixels.groupby('cluster').mean().values
    mean_image = apply_clustering(means)
    imsave('images/parrots_mean_' + str(nc) + '.png', mean_image)

    medians = pixels.groupby('cluster').median().values
    median_image = apply_clustering(medians)
    imsave('images/parrots_median_' + str(nc) + '.png', median_image)

    psnr_mean = get_psnr(norm_image, mean_image)
    psnr_median = get_psnr(norm_image, median_image)

    max_psnr = max(psnr_mean, psnr_median)

    print('mean   ' if psnr_mean > psnr_median else 'median ', nc, max_psnr)

    if max_psnr > 20 and ans == -1:
        ans = nc

print('ans =', ans)
