import numpy as np

def naive_bayes(vec, dataset, labels):
    p1 = _get_proportion(vec, 1, dataset, labels)
    p0 = _get_proportion(vec, 0, dataset, labels)
    print("Bad: %d%%\nGood: %d%%" % (p1 * 100, p0 * 100))
    return p1 > p0


def _get_proportion(vec, tar_label, dataset, labels):
    sub_dataset = []
    for i, label in enumerate(labels):
        if label == tar_label:
            sub_dataset.append(dataset[i])
    p_a = len(sub_dataset) / len(labels)
    sub_dataset = np.array(sub_dataset)
    sub_dataset_sum = np.sum(sub_dataset, axis=0)
    sub_dataset_p = sub_dataset_sum / len(sub_dataset)
    p_b_a = 1
    for i, value in enumerate(vec):
        if (value == 1):
            p_b_a *= sub_dataset_p[i]
    return p_b_a * p_a
