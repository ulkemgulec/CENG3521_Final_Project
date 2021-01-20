from glob import glob
import numpy as np
import matplotlib.pyplot as plt

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