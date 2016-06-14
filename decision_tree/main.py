import os

import numpy as np
from alg import create_decition_tree

def main():
    dataset = load_dataset("%s/%s" % (os.path.abspath(os.path.dirname(__file__)), "data/golf.txt"))
    tree = create_decition_tree(dataset)

def load_dataset(filename):
    dataset = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            fields = line.split(" ")
            dataset.append(fields)
        return np.array(dataset)

if __name__ == "__main__":
    main()
