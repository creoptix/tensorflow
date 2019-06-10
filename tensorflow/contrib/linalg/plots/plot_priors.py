import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

font = {
    'family': 'normal',
    'weight': 'bold',
    'size': 14
}

matplotlib.rc('font', **font)

n_samples = 5000000

x_normal = np.random.normal(scale=0.5, size=n_samples)
sns.kdeplot(x_normal, shade=True, label="Gaussian")

x_laplace = np.random.laplace(scale=0.5, size=n_samples)
sns.kdeplot(x_laplace, shade=True, label="Laplace")

x_uniform = np.linspace(start=-1.0, stop=1.0, num=50)
y_uniform = 0.5 * np.ones_like(x_uniform)
plt.fill_between(x_uniform, y_uniform, label="Uniform", color="g", alpha=0.25)

plt.xlabel("x")
plt.ylabel("Prior Probability")

plt.xlim([-1.25, 1.25])
plt.ylim([0.0, 1.0])
plt.legend()
plt.grid()

plt.savefig("./prior-plot.pdf")
plt.show()
