import numpy as np


class Gaussian():

    def __init__(self, mean, std, num_samples=1000):

        self.mean = mean
        self.std = std
        self.num_samples = num_samples

    def pdf(self, data):
        return (1 / np.sqrt(2 * np.pi * self.std ** 2)) * np.exp((- (data - self.mean) ** 2) / (2 * self.std ** 2))

    def log_pdf(self, data):
        return (-0.5 * (2 * np.pi)) - (0.5 * np.log(self.std ** 2)) - (((data - self.mean) ** 2) / (2 * self.std **2))

    def sample(self, num_samples):
        pass

    def fit(self, x):
        pass

