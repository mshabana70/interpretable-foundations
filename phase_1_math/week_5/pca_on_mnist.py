import math
import numpy as np
from sklearn.datasets import fetch_openml


def import_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    return mnist

def custom_pca():
    pass

def test():
    pass

if __name__ == "__main__":
    test()