import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imread

from alg import k_means

def main():
    img = load_img("%s/%s" % (os.path.abspath(os.path.dirname(__file__)), "data/test.png"))
    dataset = extract_gray_scaled_dataset(img)
    print("\n**** Beginning of K-Means ****")
    clusters = k_means(dataset)
    colors = [ "#ff0000", "#00ff00", "#0000ff" ]
    for i, cluster in enumerate(clusters):
        plt.scatter(cluster[:, 0], cluster[:, 1], c = colors[i], s = 10)
    plt.show()
    print("******* End of K-Means *******")

def extract_gray_scaled_dataset(img):
    print("Extracting points...")
    gray_scaled_img = np.average(img, axis = -1)
    points = [];
    for y in range(gray_scaled_img.shape[0]):
        row = gray_scaled_img[y]
        for x in range(row.shape[0]):
            if row[x] != 255:
                points.append((x, y))
    print("%d points were extracted from the image." % len(points))
    return np.array(points)

def load_img(file_name):
    print("Loading image '%s'..." % file_name)
    img = imread(file_name)
    print("Image loaded: %d x %d" % (img.shape[1], img.shape[0]))
    return img

if __name__ == "__main__":
    main()
