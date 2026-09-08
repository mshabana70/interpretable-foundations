import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

class Gaussian():

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def pdf(self, data):
        return ((1 / np.sqrt(2 * np.pi * self.std ** 2)) * np.exp((- (data - self.mean) ** 2) / (2 * (self.std ** 2))))[0]

    def log_pdf(self, data):
        return ((-0.5 * np.log(2 * np.pi)) - (0.5 * np.log(self.std ** 2)) - (((data - self.mean) ** 2) / (2 * self.std ** 2)))[0]

    def sample(self, num_samples):
        return rng.normal(self.mean, self.std, size=num_samples)

# TODO: Need to implement the EM algorithm for a 2-component gaussian mixture model and test on synthetic data with two clusters.
def em_algo(observations, dist_1, dist_2, pi_1, pi_2):

    # create two initialized gaussians
    mean_guess_1 = rng.normal(scale=10.0, size=1)[0]
    std_guess_1 = abs(rng.normal(scale=5.0, size=1))[0] # std needs to be positive
    weight_guess_1 = rng.random()
    init_dist_1 = Gaussian(mean_guess_1, std_guess_1)

    mean_guess_2 = rng.normal(scale=10.0, size=1)[0]
    std_guess_2 = abs(rng.normal(scale=5.0, size=1))[0]
    weight_guess_2 = 1.0 - weight_guess_1 # both weights needs to sum to 1
    init_dist_2 = Gaussian(mean_guess_2, std_guess_2)

    iter = 0
    loop_flag = True

    # loop until our param updates are small
    while loop_flag:
        # we need to make sure we reset the responsibilities table after each loop
        responsibilities_table = {"G1": [], "G2": []}
        # now we start with the E-step
        for x_i in observations:
            # compute the responsibility of x_i for G1 and G2
            pdf_1 = init_dist_1.pdf(x_i)
            pdf_2 = init_dist_2.pdf(x_i)
            prob_x_i = (weight_guess_1 * pdf_1) + (weight_guess_2 * pdf_2)

            r_i1 = float((weight_guess_1 * pdf_1) / prob_x_i)
            r_i2 = float((weight_guess_2 * pdf_2) / prob_x_i)

            # a quick test to ensure that our responsibilities are normalized
            if np.isclose(r_i1 + r_i2, 1.0, rtol=1e-5) is False:
                print(r_i1, r_i2, r_i1 + r_i2)
                print(f"[ERROR] Responsibilities don't sum to 1!!")
                return

            responsibilities_table["G1"].append(r_i1)
            responsibilities_table["G2"].append(r_i2)

        # now we calculate our updates to the dists
        resp_sum_1 = sum(responsibilities_table["G1"])
        new_mean_1 = sum([responsibilities_table["G1"][i] * observations[i] for i in range(len(observations))]) / resp_sum_1
        new_std_1 = (sum([responsibilities_table["G1"][i] * ((observations[i] - new_mean_1) ** 2) for i in range(len(observations))]) / resp_sum_1) ** 0.5
        new_weight_1 = resp_sum_1 / len(responsibilities_table["G1"])

        resp_sum_2 = sum(responsibilities_table["G2"])
        new_mean_2 = sum([responsibilities_table["G2"][i] * observations[i] for i in range(len(observations))]) / resp_sum_2
        new_std_2 = (sum([responsibilities_table["G2"][i] * ((observations[i] - new_mean_2) ** 2) for i in range(len(observations))]) / resp_sum_2) ** 0.5
        new_weight_2 = resp_sum_2 / len(responsibilities_table["G2"])

        # define the updated gaussians
        init_dist_1 = Gaussian(new_mean_1, new_std_1)
        init_dist_2 = Gaussian(new_mean_2, new_std_2)

        # check if our parameter updates converged
        if abs(new_mean_1 - mean_guess_1) <= 1e-5:
            loop_flag = False

        # set the updates
        weight_guess_1 = new_weight_1
        mean_guess_1 = new_mean_1
        std_guess_1 = new_std_1

        weight_guess_2 = new_weight_2
        mean_guess_2 = new_mean_2
        std_guess_2 = new_std_2

        iter += 1
        print(f"Iteration {iter} complete | means = [{mean_guess_1, mean_guess_2}] | stds = means = [{std_guess_1, std_guess_2}] | weights = [{weight_guess_1, weight_guess_2}]")

    # print how close we got with our updates to the true dist parameters
    print(f"===== EM Algorithm Completed in {iter} iterations ====")
    print(f"True G1 Mean: {dist_1.mean} | Estimated G1 Mean: {mean_guess_1}")
    print(f"True G1 Std: {dist_1.std} | Estimated G1 Std: {std_guess_1}")
    print(f"True G1 mixture weight: {pi_1} | Estimated G1 mixture weight: {weight_guess_1}")

    print(f"True G2 Mean: {dist_2.mean} | Estimated G1 Mean: {mean_guess_2}")
    print(f"True G2 Std: {dist_2.std} | Estimated G1 Std: {std_guess_2}")
    print(f"True G2 mixture weight: {pi_2} | Estimated G1 mixture weight: {weight_guess_2}")
        

def test():

    sample_size = 10000

    # just creating some random dists
    norm_dist_1 = Gaussian(5.0, 2.0)
    norm_dist_2 = Gaussian(3.0, 0.5)

    selector = [1, 2]
    obv_dataset = []
    pi_1, pi_2 = 0.6, 0.4
    for _ in range(sample_size):
        pick = np.random.choice(selector, p=[pi_1, pi_2])
        if pick == 1:
            obv_dataset.append(norm_dist_1.sample(1))
        else:
            obv_dataset.append(norm_dist_2.sample(1))

    results = []

    em_algo(obv_dataset, norm_dist_1, norm_dist_2, pi_1, pi_2)


if __name__ == "__main__":
    test()