import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

class Gaussian():

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def pdf(self, data):
        return (1 / np.sqrt(2 * np.pi * self.std ** 2)) * np.exp((- (data - self.mean) ** 2) / (2 * (self.std ** 2)))

    def log_pdf(self, data):
        return (-0.5 * np.log(2 * np.pi)) - (0.5 * np.log(self.std ** 2)) - (((data - self.mean) ** 2) / (2 * self.std ** 2))

    def sample(self, num_samples):
        return rng.normal(self.mean, self.std, size=num_samples)

    def fit(self, x):
        mle_mean = np.mean(x)
        mle_std = np.std(x)
        return mle_mean - mle_std

# TODO: Need to implement the EM algorithm for a 2-component gaussian mixture model and test on synthetic data with two clusters.


if __name__ == "__main__":
    print(f"Need to write EM algo for 2-component gaussian mixture model from scratch.")