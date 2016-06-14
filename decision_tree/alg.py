import numpy as np
import math

from node import *

def create_decition_tree(dataset):
    root = __create_branch(dataset)
    return root


def classify(data, node):
    if node.type == "result":
        return node.value
    elif node.type == "feature":
        value = data[node.value]
        for value_node in node.children:
            if value == value_node.value:
                return classify(data, value_node)
    elif node.type == "value":
        feature_node = node.children[0]
        return classify(data, feature_node)
    return None


def calc_entropy(dataset):
    labels = dataset[:, -1]
    labelDict = {}
    for label in labels:
        labelDict[label] = labelDict.get(label, 0) + 1
    entropy = 0
    for item in labelDict.items():
        p = item[1] / len(labels)
        entropy += -p * math.log(p, 2)
    return entropy





def __create_branch(dataset):
    split_result = __choose_best_feature_and_split(dataset)
    feature_node = FeatureNode(split_result["feature_index"])
    for value, dataset in split_result["datasets"].items():
        value_node = ValueNode(value)
        entropy = calc_entropy(dataset)
        if (entropy > 0):
            value_node.append_child(__create_branch(dataset))
        else:
            value_node.append_child(ResultNode(dataset[0, -1]))
        feature_node.append_child(value_node)
    return feature_node


def __choose_best_feature_and_split(dataset):
    result = None
    for i in range(dataset.shape[-1] - 1):
        splitted_datasets = __split_dataset(dataset, i)
        entropy = __calc_avg_entropy(splitted_datasets.values())
        if result == None or result["entropy"] > entropy:
            result = {
                "feature_index": i,
                "entropy": entropy,
                "datasets": splitted_datasets
            }
    return result


def __split_dataset(dataset, feature_index):
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


def __calc_avg_entropy(datasets):
    count = 0
    for dataset in datasets:
        count += len(dataset)
    avg = 0
    for dataset in datasets:
        weight = len(dataset) / count
        avg += weight * calc_entropy(dataset)
    return avg
