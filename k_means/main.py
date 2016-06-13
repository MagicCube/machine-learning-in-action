import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imread

from alg import k_means

def main():
    img = load_img("%s/%s" % (os.path.abspath(os.path.dirname(__file__)), "data/test.png"))
    dataset = extract_rgb_dataset(img)
    print("\n**** Beginning of K-Means ****")
    clusters = k_means(5, dataset)
    colors = [ "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#e377c2" ]
    markers = [ 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd' ]
    for i, cluster in enumerate(clusters):
        print("#%d: %d" % (i, len(cluster)))
        plt.scatter(cluster[:, 0], cluster[:, 1] * -1, s=15, c=colors[i], marker = markers[i])
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

def extract_rgb_dataset(img):
    print("Extracting points...")
    points = [];
    height, width, color_depth = img.shape
    for y in range(height):
        row = img[y]
        for x in range(width):
            color = row[x];
            if not np.array_equal(color, [255, 255, 255]):
                points.append((x / width, y / height, color[0] / 255, color[1] / 255, color[2] / 255))
    print("%d points were extracted from the image." % len(points))
    return np.array(points)

def load_img(file_name):
    print("Loading image '%s'..." % file_name)
    img = imread(file_name)
    print("Image loaded: %d x %d" % (img.shape[1], img.shape[0]))
    return img

if __name__ == "__main__":
    main()
