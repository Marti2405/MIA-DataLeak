import matplotlib.pyplot as plt
import numpy as np
from statistics import NormalDist
import seaborn as sn


class Gaussian:
    dist: NormalDist

    def __init__(self, mean, std):
        self.dist = NormalDist(mean, std)

    def hist(self):
        sample = np.random.normal(self.dist.mean, self.dist.stdev, 500)
        sn.kdeplot(sample)

    def plot(self):
        self.hist()
        plt.show()

    def compare(self, other, x=None):
        self.hist()
        other.hist()
        if x:
            plt.axvline(x=x, color="red", linestyle="--", alpha=0.5)
        plt.show()

    def z_score(self, x: float):
        return self.dist.zscore(x)

    def cdf(self, x):
        return self.dist.cdf(x)

    def pdf(self, x):
        return self.dist.pdf(x)

    def __repr__(self) -> str:
        return f"(mu={self.dist.mean}, sigma={self.dist.stdev})"


if __name__ == "__main__":
    my_gaussian = Gaussian(1, 2)
    my_gaussian_dos = Gaussian(12, 1)
    print(my_gaussian.z_score(12))
    print(my_gaussian.cdf(1))
    print(my_gaussian.pdf(12))
    print(my_gaussian_dos.pdf(12))
    my_gaussian_dos.compare(my_gaussian, 12)
