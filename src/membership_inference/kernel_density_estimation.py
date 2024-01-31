import seaborn as sn
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


class KDE:
    def __init__(self, data):
        self.density = gaussian_kde(data)

    def kde(self):
        sn.kdeplot(self.density)

    def plot(self):
        self.kde()
        plt.show()

    def get_density(self):
        return self.density    
