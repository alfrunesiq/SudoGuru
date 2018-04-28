import os
import sys
import cv2 as cv
import numpy as np
import zipfile

if len(sys.argv) < 2:
    print ("Usage: python convert_dataset.py /path/to/dataset")
    sys.exit(0)

images = []
idx    = [0]
path_base = sys.argv[1]
path_suffix = []
for i in range(2,11):
    path_suffix.append("Sample0%02d" % i)

print ("Loading and resizing images...")
for suffix in path_suffix:
    path = os.path.join(path_base, suffix)
    for file in os.listdir(path):
        if file.endswith(".png"):
            img = cv.imread(os.path.join(path, file))
            img = cv.resize(img, (32, 32))
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            imgf = img.astype(np.float32)
            imgf = 1.0 - (imgf - np.min(imgf))/(np.max(imgf) - np.min(imgf))
            images.append(imgf)
    idx.append(len(images))
y_labels = np.zeros(idx[-1], dtype=np.int)
for i in range(1, len(idx)):
    print("[%d, %d]" % (idx[i-1], idx[i]))
    y_labels[idx[i-1]:idx[i]] = i
images = np.array(images)

# Shuffle
idx      = np.arange(images.shape[0])
idx      = np.random.choice(idx, idx.shape[0],
                            replace=False)
images   = images[idx,:,:]
y_labels = y_labels[idx]

print ("Saving images to /tmp/digits_[img|lbl].npy")
images = np.array(images)
np.save("/tmp/digits_img", images)
np.save("/tmp/digits_lbl", y_labels)

