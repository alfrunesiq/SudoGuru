import os
import sys
import requests
import tarfile

import cv2 as cv
import numpy as np
import re

url = "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz"

images = []
idx    = [0]

if not os.path.isdir("/tmp"):
    # Might be the case for all ya windows folks
    os.mkdir("/tmp")

path_base = "/tmp/English/Fnt/"
path_suffix = []
for i in range(2,11):
    path_suffix.append("Sample0%02d" % i)

if not os.path.isdir(path_base):
    # Download dataset:
    print("Downloading imageset...")
    r = requests.get(url)
    if r.status_code != 200:
        print("ERROR: Failed to download dataset")
        sys.exit(1)

    with open("/tmp/chars74k.tgz", "wb")as f:
        f.write(r.content)
    os.makedirs(path_base)
    print("Extracting content...")
    with tarfile.open("/tmp/chars74k.tgz", "r") as tar:
        # Only extract 1-9
        for path in tar.getnames():
            if not path.endswith(".png"):
                for suffix in path_suffix:
                    if path.endswith(suffix):
                        tar.extract(path, "/tmp/")


print ("Loading and resizing images...")
for suffix in path_suffix:
    path = os.path.join(path_base, suffix)
    for file in os.listdir(path):
        if file.endswith(".png"):
            r = re.search("[0-9]{5}", file)
            if r == None:
                print(file)
                continue
            num = int(r.group(0))
            if (num % 4 < 3 and num % 4 != 0):
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

print ("Saving images to /tmp/English/digits_[img|lbl].npy")
images = np.array(images)
np.save("/tmp/English/digits_img", images)
np.save("/tmp/English/digits_lbl", y_labels)
