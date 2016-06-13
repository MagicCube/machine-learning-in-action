import numpy as np

def knn(k, training_dataset, training_labels, x):
    distance = np.sqrt(np.sum((training_dataset - x) ** 2, axis=-1))
    sortedIndex = np.argsort(distance)
    labels = {}
    for i in range(k):
        index = sortedIndex[i]
        label = training_labels[index]
        labels[label] = labels.get(label, 0) + 1
    sortedList = sorted(labels.items(), key=lambda item: item[1], reverse=True)
    return sortedList[0][0]
