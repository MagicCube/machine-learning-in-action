import numpy as np
import math

from node import *

def create_decition_tree(dataset):
    root = create_branch(dataset)
    print(root)


def create_branch(dataset):
    split_result = choose_best_feature_and_split(dataset)
    feature_node = FeatureNode(split_result["feature_index"])
    for value, dataset in split_result["datasets"].items():
        value_node = ValueNode(value)
        entropy = _calc_entropy(dataset)
        if (entropy > 0):
            value_node.append_child(create_branch(dataset))
        else:
            value_node.append_child(ResultNode(dataset[0, -1]))
        feature_node.append_child(value_node)
    return feature_node


def choose_best_feature_and_split(dataset):
    result = None
    for i in range(dataset.shape[-1] - 1):
        splitted_datasets = split_dataset(dataset, i)
        entropy = _calc_avg_entropy(splitted_datasets.values())
        if result == None or result["entropy"] > entropy:
            result = {
                "feature_index": i,
                "entropy": entropy,
                "datasets": splitted_datasets
            }
    return result


def split_dataset(dataset, feature_index):
    datasetDict = {}
    for row in dataset:
        value = row[feature_index]
        if value not in datasetDict:
            datasetDict[value] = []
        datasetDict[value].append(row)
    result = {}
    for key, dataset in datasetDict.items():
        result[key] = np.array(dataset)
    return result


def _calc_avg_entropy(datasets):
    count = 0
    for dataset in datasets:
        count += len(dataset)
    avg = 0
    for dataset in datasets:
        weight = len(dataset) / count
        avg += weight * _calc_entropy(dataset)
    return avg

def _calc_entropy(dataset):
    labels = dataset[:, -1]
    labelDict = {}
    for label in labels:
        labelDict[label] = labelDict.get(label, 0) + 1
    entropy = 0
    for item in labelDict.items():
        p = item[1] / len(labels)
        entropy += -p * math.log(p, 2)
    return entropy
