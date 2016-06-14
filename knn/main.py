import os
import numpy as np

from alg import knn

def main():
    K = 5
    TRAINING_RATE = 0.9

    dataset, labels = load_data("%s/%s" % (os.path.abspath(os.path.dirname(__file__)), "data/datingTestSet.txt"))
    dataset = normalize(dataset)
    # Picking up samples
    training_count = int(len(dataset) * TRAINING_RATE)
    training_dataset, training_labels = dataset[:training_count, :], labels[:training_count]
    validation_dataset, validation_labels = dataset[training_count:, :], labels[training_count:]
    # Validating
    error_count = 0.0
    for i, validation_data in enumerate(validation_dataset):
        label = knn(K, training_dataset, training_labels, validation_data)
        if label != validation_labels[i]:
            error_count += 1
    error_rate = error_count / validation_dataset.shape[0];
    print("Errors %d / %d\nError rate: %f%%" % (error_count, validation_dataset.shape[0], error_rate * 100))


def load_data(filename):
    dataset = []
    labels = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            fields = line.split("\t")
            dataset.append(fields[0:3])
            labels.append(fields[3])
    dataset = np.array(dataset, dtype=np.float)
    return (dataset, labels)

def normalize(dataset):
    return (dataset - np.min(dataset, axis=0)) / (np.max(dataset, axis=0) - np.min(dataset, axis=0))

if __name__ == "__main__":
    main()
