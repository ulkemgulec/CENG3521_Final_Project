from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.color import rgb2grey
from sklearn.feature_extraction import image
from sklearn.cluster import KMeans
from skimage.filters import rank, threshold_otsu
from skimage.morphology import closing, square, disk
from skimage import exposure as hist, data, img_as_float
from skimage.segmentation import chan_vese
from skimage.feature import canny
from skimage.color import rgb2gray
from scipy import ndimage as ndi

#We assign malignant image data to mal_images and benign image data to ben_images.

mal_images = glob('skin_canser/data/train/malignant/*')
ben_images = glob('skin_canser/data/train/benign/*')

#Also, we assign just five images to display how data looks like.

mal_images_1 = glob('skin_canser/data/train/malignant/*')[:5]
ben_images_1 = glob('skin_canser/data/train/benign/*')[:5]

#Length of the malignant images. This is the number of malignant images.

len(mal_images)

#Length of the benign images. This is the number of benign images.

len(ben_images)

# These are the functions that we are using later.


# We apply  thresold_otsu alghorithm to images. That alghorithm separates pixels into two classes, foreground and background.
def binary(image):
    return image > threshold_otsu(image)


# This function returns the greyscale local mean of an image.
def mean_filter(image, radius_disk):
    return rank.mean_percentile(image, selem=disk(radius_disk))


# It loads the images from the local directory.
def load_images(paths):
    tmp = []
    for path in paths:
        tmp.append(imread(path))
    return tmp


# It displays the graphics or photographs as subplots or plot.
def plot_any(arr, title=''):
    plt.figure(figsize=(15, 25))
    for i in range(len(arr)):
        plt.subplot(1, len(arr), i + 1)
        plt.title(title)
        plt.imshow(arr[i]);


# This function applies the KMeans clustering algorithm.
def d2Kmeans(img, k):
    return KMeans(n_jobs=-1,
                  random_state=1,
                  n_clusters=k,
                  init='k-means++'
                  ).fit(img.reshape((-1, 1))).labels_.reshape(img.shape)


# This function applies a mask to a given image.
def merge_segmented_mask_ROI(uri_img, img_kluster):
    new_img = uri_img.copy()
    for ch in range(3):
        new_img[:, :, ch] *= img_kluster
    return new_img


# This function applies the KMeans algorithm more than one time that algorithm shows us elbow.
def elbow(img, k):
    hist = []
    for kclusters in range(1, k):
        Km = KMeans(n_jobs=-1, random_state=1, n_clusters=kclusters, init='k-means++').fit(img.reshape((-1, 1)))
        hist.append(Km.inertia_)

    plt.figure(figsize=(15, 8))
    plt.grid()
    plt.plot(range(1, k), hist, 'o-')
    plt.ylabel('Sum of squared distances')
    plt.xlabel('k clusters')
    plt.title('Elbow')
    plt.show();

#We assign mal_images to mal using load_images function.
#We assign ben_images to ben using load_images function.

mal = load_images(mal_images)
ben = load_images(ben_images)

mal_1 = load_images(mal_images_1)
ben_1 = load_images(ben_images_1)

#There are five images example for each moles type.

plot_any(mal_1, "Malignant Data")
plot_any(ben_1, "Benign Data")


#We pick a photograph from mal images to apply the elbow algorithm.

img_selected = mal[2]

elbow(img_selected, 6)

#We find the elbow from the elbow graph and we assign that number to the k_cluster variable to using later.
k_klusters = 2

#We put the selected image into the d2Kmean function twice with the k_cluster number. One of them makes it 2 colors, another one is original

result_gray = d2Kmeans(rgb2grey(img_selected), k_klusters)
result_img = d2Kmeans(img_selected, k_klusters)

#We show the two color image with the different k_cluster numbers.

klusters_gray = [result_gray == i for i in range (k_klusters)]
plot_any(klusters_gray)

#This function helps us to pick a suitable image

def select_cluster_index(clusters):
    minx = clusters[0].mean()
    index = 0
    for i in clusters:
        if i.mean() < minx:
            minx = i.mean()
            index += 1
    return index

#We assign the image selected from select_cluster_index function

index_kluster = select_cluster_index(klusters_gray)
print(index_kluster)
selected = klusters_gray[index_kluster]

# We display the images of k_clusters number how it affects to image.

for ch in range(3):
    img_k = []
    for K in range(k_klusters):
        img_k.append(result_img[:, :, ch] == K)
    plot_any(img_k)

clusters = [(result_img[:,:,1] == K) for K in range(k_klusters)]

clusters

#We apply mask to a selected image.

new_img = merge_segmented_mask_ROI(img_selected, selected)

#We display masked image.

plot_any([new_img])

#We choose 20 as radius disk and we give with selected image to mean filter function.
image_mean_filter = mean_filter(selected, 20)

#Image means filtered image convert to binary form.
test_binary = binary(image_mean_filter)

plot_any([selected, image_mean_filter, test_binary])

#Image's original form with the mask.

final_result = merge_segmented_mask_ROI(img_selected ,test_binary)

final_result.shape

plot_any([test_binary, new_img, final_result])



