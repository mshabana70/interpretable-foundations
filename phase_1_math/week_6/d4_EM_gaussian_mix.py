import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

class Gaussian():

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def pdf(self, data):
        return (1 / np.sqrt(2 * np.pi * self.std ** 2)) * np.exp((- (data - self.mean) ** 2) / (2 * (self.std ** 2)))

    

if __name__ == "__main__":
    print(f"Need to write EM algo for 2-component gaussian mixture model from scratch.")