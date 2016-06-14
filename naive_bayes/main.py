import os
import re

import numpy as np

from alg import naive_bayes

def main():
    dataset, labels, dict = load_dataset("%s/%s" % (os.path.abspath(os.path.dirname(__file__)), "data/post.txt"))

    text = "Stop posting stupid worthless garbage!"
    print(text)
    result = naive_bayes(convert_to_vector(text, dict), dataset, labels)
    print("Banned: ", result)

    print()

    text = "Mr Licks ate my steak, How to stop him!"
    print(text)
    result = naive_bayes(convert_to_vector(text, dict), dataset, labels)
    print("Banned: ", result)


def load_dataset(filename):
    posts, labels, dict = load_posts(filename)
    dataset = []
    for post in posts:
        vector = []
        dataset.append(vector)
        for word in dict:
            if word in post:
                vector.append(1)
            else:
                vector.append(0)
    return (
        np.array(dataset),
        labels,
        dict
    )


def convert_to_vector(text, dict):
    reg_exp = re.compile(r'(\w+)')
    post = list(map(lambda word: word.lower(), reg_exp.findall(text)))
    vector = []
    for word in dict:
        if word in post:
            vector.append(1)
        else:
            vector.append(0)
    return vector


def load_posts(filename):
    reg_exp = re.compile(r'(\w+)')
    posts = []
    labels = []
    dict = set()
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            label = int(line[0] == "-")
            text = line[2:]
            words = list(map(lambda word: word.lower(), reg_exp.findall(text)))
            dict = dict.union(words)
            posts.append(words)
            labels.append(label)
    return (
        posts,
        np.array(labels),
        list(dict)
    )

if __name__ == "__main__":
    main()
