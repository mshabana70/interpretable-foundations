import math
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

class Gaussian():

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def pdf(self, data):
        return (1 / np.sqrt(2 * np.pi * self.std ** 2)) * np.exp((- (data - self.mean) ** 2) / (2 * self.std ** 2))

    def log_pdf(self, data):
        return (-0.5 * np.log(2 * np.pi)) - (0.5 * np.log(self.std ** 2)) - (((data - self.mean) ** 2) / (2 * self.std **2))

    def sample(self, num_samples):
        return rng.normal(self.mean, self.std, size=num_samples)

    def fit(self, x):
        mle_mean = np.mean(x)
        mle_std = np.std(x)
        return mle_mean, mle_std


def test():

    sample_size = 10000
    norm_dist_1 = Gaussian(5.0, 2.0)
    norm_dist_1_samples = norm_dist_1.sample(sample_size)

    # verify samples are from norm_dist_1 distribution:
    dist_1_ver_mean, dist_1_ver_std = norm_dist_1.fit(norm_dist_1_samples)
    if math.isclose(dist_1_ver_mean, norm_dist_1.mean, abs_tol=0.25) and math.isclose(dist_1_ver_std, norm_dist_1.std, abs_tol=0.25):
        print(f"Distribution 1 samples are verified!!")
    else:
        print(f"Distribution 1 samples failed verification!!")
    
    print(f"MLE mean: {dist_1_ver_mean}, True mean: {norm_dist_1.mean}")
    print(f"MLE std: {dist_1_ver_std}, True std: {norm_dist_1.std}")

    norm_dist_2 = Gaussian(3.0, 0.5)
    norm_dist_2_samples = norm_dist_2.sample(sample_size)

    # verify samples are from norm_dist_1 distribution:
    dist_2_ver_mean, dist_2_ver_std = norm_dist_2.fit(norm_dist_2_samples)
    if math.isclose(dist_2_ver_mean, norm_dist_2.mean, abs_tol=0.25) and math.isclose(dist_2_ver_std, norm_dist_2.std, abs_tol=0.25):
        print(f"Distribution 2 samples are verified!!")
    else:
        print(f"Distribution 2 samples failed verification!!")
    
    print(f"MLE mean: {dist_2_ver_mean}, True mean: {norm_dist_2.mean}")
    print(f"MLE std: {dist_2_ver_std}, True std: {norm_dist_2.std}")

    # now we can do some plotting
    x_ax = np.linspace(-15, 15, sample_size)

    y1 = norm_dist_1.pdf(x_ax)
    y2 = norm_dist_2.pdf(x_ax)
    # let's try ploting a third
    y3 = Gaussian(2.0, 5.0).pdf(x_ax)

    plt.figure(figsize=(8, 5))
    plt.plot(x_ax, y1, label=r'$\mu=5.0, \sigma=2.0$', color='blue', linewidth=2)
    plt.plot(x_ax, y2, label=r'$\mu=3.0, \sigma=0.5$', color='red', linewidth=2)
    plt.plot(x_ax, y3, label=r'$\mu=2.0, \sigma=5.0$', color='green', linewidth=2)

    plt.title('Two Gaussian Distributions')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("./figs/d3_two_normal_dists.png")

if __name__ == "__main__":
    test()

