import math
import numpy as np
import random

def mle_mean(samples):
    n = len(samples)
    return sum(samples) / n

def standard_deviation(samples, mean):
    n = len(samples)
    diff_sum = sum([(sample - mean) ** 2 for sample in samples])
    return math.sqrt(((1/n) * diff_sum))


def MLE(samples):
    # since we did most of the math in our journal, we can just jump to the 
    # resulting equations for estimating the distribution parameters sigma and mu:

    # first we call the mean:
    mean = mle_mean(samples)
    std = standard_deviation(samples, mean)
    return mean, std

def test():

    # we are going to generate random datasets here with predefined means and stds
    pre_mean = 0
    pre_std = 2
    pre_size = 1000
    test1_array = np.random.normal(loc=pre_mean, scale=pre_std, size=pre_size)
    mean_1, std_1 = MLE(test1_array)
    print(10*"=", "Test 1", 10*"=")
    print(f"Distribution Mean is {pre_mean} and Standard Deviation is {pre_std}")
    print(f"My mean: {mean_1}, NumPy Mean: {np.mean(test1_array)}")
    print(f"My standard dev: {std_1}, NumPy standard dev: {np.std(test1_array)}\n")

    pre_mean = 23
    pre_std = 4
    pre_size = 1000
    test2_array = np.random.normal(loc=pre_mean, scale=pre_std, size=pre_size)
    mean_2, std_2 = MLE(test2_array)
    print(10*"=", "Test 2", 10*"=")
    print(f"Distribution Mean is {pre_mean} and Standard Deviation is {pre_std}")
    print(f"My mean: {mean_2}, NumPy Mean: {np.mean(test2_array)}")
    print(f"My standard dev: {std_2}, NumPy standard dev: {np.std(test2_array)}")

if __name__ == "__main__":
    test()