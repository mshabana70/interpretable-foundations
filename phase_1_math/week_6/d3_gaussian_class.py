import math
import numpy as np
import matplotlib


# for wsl execution
matplotlib.use("Agg")

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


def check(name, passed, detail=""):
    """Print one PASS/FAIL line and return the boolean so the runner can tally."""
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))
    return bool(passed)


def test_fit_recovers_parameters(mean, std, n=10_000, k=4.0):
    """Acceptance test 1: sample() then fit() should recover the parameters we started with.

    The tolerance is STATISTICAL, so it must be built from sigma and N -- not hardcoded:
        SE(mu_hat)    = sigma / sqrt(N)
        SE(sigma_hat) = sigma / sqrt(2N)        <- tighter than the mean's by sqrt(2)
    k = 4 standard errors => roughly a 1-in-16,000 false-failure rate per assertion,
    while still catching a sampler whose parameters are genuinely off.
    """
    g = Gaussian(mean, std)
    mle_mean, mle_std = g.fit(g.sample(n))

    tol_mean = k * std / np.sqrt(n)
    tol_std = k * std / np.sqrt(2 * n)

    ok_mean = check(
        f"fit recovers mu={mean} from {n} samples",
        math.isclose(mle_mean, mean, abs_tol=tol_mean),
        f"mu_hat={mle_mean:.5f}, err={abs(mle_mean - mean):.5f}, tol={tol_mean:.5f}",
    )
    ok_std = check(
        f"fit recovers sigma={std} from {n} samples",
        math.isclose(mle_std, std, abs_tol=tol_std),
        f"sigma_hat={mle_std:.5f}, err={abs(mle_std - std):.5f}, tol={tol_std:.5f}",
    )
    return ok_mean and ok_std


def test_pdf_integrates_to_one(mean, std, span=12.0, n_grid=200_001):
    """Acceptance test 2: the density must integrate to 1 over a wide grid.

    This one is a DETERMINISTIC numerical integral, not an estimate, so np.isclose's
    default tolerances (rtol=1e-5) are appropriate here -- note how different that is
    from test 1's tolerance, despite both being "is this close to the right answer".
    """
    g = Gaussian(mean, std)
    grid = np.linspace(mean - span * std, mean + span * std, n_grid)
    area = np.trapezoid(g.pdf(grid), grid)
    return check(
        f"pdf integrates to 1 over mu +/- {span:g}sigma",
        np.isclose(area, 1.0),
        f"area={area:.10f}",
    )


def test_log_pdf_is_true_log_space(mean, std, n_sigma_far=40):
    """Acceptance test 3: log_pdf must be computed in log-space, not as log(pdf(x)).

    Two halves, and BOTH are needed:
      (a) it agrees with log(pdf(x)) wherever pdf(x) is safely representable
          -> proves the algebra is right, not just finite
      (b) it stays finite far into the tail where pdf(x) has underflowed to 0.0
          -> proves it never materialises exp(-large)
    """
    g = Gaussian(mean, std)

    # (a) safe region: stay within a few sigma so pdf(x) is nowhere near underflow
    safe = np.linspace(mean - 5 * std, mean + 5 * std, 1001)
    max_err = float(np.max(np.abs(g.log_pdf(safe) - np.log(g.pdf(safe)))))
    ok_agree = check(
        "log_pdf agrees with log(pdf) in the safe region",
        max_err < 1e-12,
        f"max abs err={max_err:.3e}",
    )

    # (b) far tail: exponent is -(n_sigma_far**2)/2, well past float64's exp underflow (~-744)
    far = mean + n_sigma_far * std
    with np.errstate(divide="ignore"):
        naive = float(np.log(g.pdf(far)))
    lp = float(g.log_pdf(far))

    ok_underflow = check(
        f"pdf underflows to exactly 0 at mu+{n_sigma_far}sigma (so log(pdf) is unusable)",
        g.pdf(far) == 0.0 and not np.isfinite(naive),
        f"pdf={g.pdf(far)}, log(pdf)={naive}",
    )
    ok_finite = check(
        f"log_pdf stays finite at mu+{n_sigma_far}sigma",
        np.isfinite(lp),
        f"log_pdf={lp:.6f}",
    )
    return ok_agree and ok_underflow and ok_finite


def test():

    sample_size = 10000
    norm_dist_1 = Gaussian(5.0, 2.0)
    norm_dist_2 = Gaussian(3.0, 0.5)

    results = []

    print("Test 1 -- sample/fit round trip")
    results.append(test_fit_recovers_parameters(norm_dist_1.mean, norm_dist_1.std, n=sample_size))
    results.append(test_fit_recovers_parameters(norm_dist_2.mean, norm_dist_2.std, n=sample_size))
    # mu = 0 is the case a relative tolerance can never handle -- worth keeping in the suite
    results.append(test_fit_recovers_parameters(0.0, 1.0, n=sample_size))

    print("\nTest 2 -- pdf normalisation")
    results.append(test_pdf_integrates_to_one(norm_dist_1.mean, norm_dist_1.std))
    results.append(test_pdf_integrates_to_one(norm_dist_2.mean, norm_dist_2.std))

    print("\nTest 3 -- log_pdf is genuine log-space")
    results.append(test_log_pdf_is_true_log_space(norm_dist_1.mean, norm_dist_1.std))
    results.append(test_log_pdf_is_true_log_space(norm_dist_2.mean, norm_dist_2.std))

    print(f"\n{sum(results)}/{len(results)} checks passed.\n")

    # now we can do some plotting
    # grid resolution is a rendering choice -- keep it separate from sample_size
    x_ax = np.linspace(-15, 15, 1000)

    y1 = norm_dist_1.pdf(x_ax)
    y2 = norm_dist_2.pdf(x_ax)
    # let's try ploting a third
    y3 = Gaussian(2.0, 5.0).pdf(x_ax)

    mixture_y1_y2 = (0.5 * y1) + (0.5 * y2) # needs to be probabilities that sum to 1

    plt.figure(figsize=(8, 5))
    plt.plot(x_ax, y1, label=r'$\mu=5.0, \sigma=2.0$', color='blue', linewidth=2)
    plt.plot(x_ax, y2, label=r'$\mu=3.0, \sigma=0.5$', color='red', linewidth=2)
    plt.plot(x_ax, y3, label=r'$\mu=2.0, \sigma=5.0$', color='green', linewidth=2)

    plt.title('Three Gaussian Distributions')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("./figs/d3_two_normal_dists.png")

if __name__ == "__main__":
    test()

