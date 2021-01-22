from glob import glob

import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import numpy as np

import matplotlib.pyplot as plt

import skimage
from skimage.io import imread
from skimage.color import rgb2grey
from sklearn.feature_extraction import image
from sklearn.cluster import KMeans
from skimage.filters import rank, threshold_otsu
from skimage.morphology import disk

import os

import cv2

from mpl_toolkits.mplot3d import Axes3D

import mahotas

import warnings
warnings.filterwarnings('ignore')




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
        plt.imshow(arr[i]);
    plt.savefig(title + ".png")
    plt.close()


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
    plt.savefig("elbow.png")
    plt.close()


#We assign mal_images to mal using load_images function.
#We assign ben_images to ben using load_images function.

mal = load_images(mal_images)
ben = load_images(ben_images)

mal_1 = load_images(mal_images_1)
ben_1 = load_images(ben_images_1)

#There are five images example for each moles type.

plot_any(mal_1, "MalignantData")
plot_any(ben_1, "BenignData")


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
plot_any(klusters_gray, "converted2color")

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
selected = klusters_gray[index_kluster]

# We display the images of k_clusters number how it affects to image.

for ch in range(3):
    img_k = []
    for K in range(k_klusters):
        img_k.append(result_img[:, :, ch] == K)
    plot_any(img_k)

clusters = [(result_img[:,:,1] == K) for K in range(k_klusters)]



#We apply mask to a selected image.

new_img = merge_segmented_mask_ROI(img_selected, selected)

#We display masked image.

plot_any([new_img], "maskedImage")

#We choose 20 as radius disk and we give with selected image to mean filter function.
image_mean_filter = mean_filter(selected, 20)

#Image means filtered image convert to binary form.
test_binary = binary(image_mean_filter)

plot_any([selected, image_mean_filter, test_binary], "Normal_MeanFiltered_Binary")

#Image's original form with the mask.

final_result = merge_segmented_mask_ROI(img_selected ,test_binary)

final_result.shape

plot_any([test_binary, new_img, final_result],  "Binary_Masked_FinalImage")



img = cv2.imread(mal_images[2])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
r, g, b = cv2.split(img)
r = r.flatten()
g = g.flatten()
b = b.flatten()
#plotting
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(r, g, b)
plt.savefig("RgbColorOfSelectedImage.png")
plt.close()

#We put masked malignant images into data_mal list.

data_mal = list()
for img in mal :
    img = merge_segmented_mask_ROI(img ,test_binary)
    data_mal.append(img)


#We put masked benign images into data_mal list.

data_ben = list()
for img in ben :
    img = merge_segmented_mask_ROI(img ,test_binary)
    data_ben.append(img)


len(data_ben)
len(data_mal)


# create a directory in which to store cropped images
out_dir = "segmented/ben/"
print("[STATUS] Masked benign images are saving in the directory ---> " + out_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# save each cropped image by its index number
for k,im in enumerate(data_ben):
    skimage.io.imsave(out_dir +'ben' + str(k) + ".jpg", im)


#This part save masked beningn images into a given path.

out_dir = "segmented/mal/"
print("[STATUS] Masked malignant images are saving in the directory ---> " + out_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# save each cropped image by its index number
for c,image in enumerate(data_mal):
    skimage.io.imsave(out_dir +'mal' + str(c) + ".jpg", image)
    
    
    
    # make a fix file size
fixed_size  = tuple((1000,1000))

#train path
train_path = "segmented/"

# no of trees for Random Forests
num_tree = 100

# bins for histograms
bins = 8

# train_test_split size
test_size = 0.10

# seed for reproducing same result
seed = 9

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    haralic = mahotas.features.haralick(gray).mean(axis=0)
    return haralic


def fd_histogram(image, mask=None):
    # convert the image to HSV colors-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #COMPUTE THE COLOR HISTOGRAM
    hist  = cv2.calcHist([image],[0,1,2],None,[bins,bins,bins], 
                         [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()
    
# get the training data labels
train_labels = os.listdir(train_path)

# sort the training labesl
train_labels.sort()
print(train_labels)

# empty list to hold feature vectors and labels
global_features = []
labels = []

i, j = 0, 0
k = 0

# num of images per class
images_per_class = 80


# lop over the training data sub folder
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    k = 1
    # loop over the images in each sub-folder   

for file in os.listdir(dir):

        file = dir + "/" + os.fsdecode(file)

        # read the image and resize it to a fixed-size
        image = cv2.imread(file)

        if image is not None:
            image = cv2.resize(image, fixed_size)
            fv_hu_moments = fd_hu_moments(image)
            fv_haralick = fd_haralick(image)
            fv_histogram = fd_histogram(image)

        # Concatenate global features
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

        i += 1
        k += 1
    print("[STATUS] processed folder: {}".format(current_label))
    j += 1

print("[STATUS] completed Global Feature Extraction...")



# get the overall feature vector size
print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
print("[STATUS] training Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)


print("[STATUS] training labels encoded...{}")
# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized...")

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))

global_features = np.array(rescaled_features)
global_labels = np.array(target)
