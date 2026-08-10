from scipy import stats
import numpy as np
from matplotlib import pyplot as plt

print(stats.binom.pmf(k=3, n=5, p=0.5))

x = np.linspace(-4, 4, 200)
plt.plot(x, stats.norm.pdf(x, loc=0, scale=1))
plt.title('Normal Distribution')
plt.savefig('normal_distribution.png')
plt.show()

np.random.seed(42)
population = 100_000
has_disease = np.random.rand(population) < 0.001
test_positive = np.where(
    has_disease,
    np.random.rand(population) < 0.99,
    np.random.rand(population) < 0.01,
)
probability_disease_given_positive = has_disease[test_positive].mean()
print("Bayes theorem:-", probability_disease_given_positive)

data = np.array([2, 4, 4, 4, 5, 5, 7, 9])
print("Mean:-", np.mean(data))
print("Median:-", np.median(data))
print("Standard Deviation:-", np.std(data))
print("Variance:-", np.var(data))