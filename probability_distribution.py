from scipy import stats
import numpy as np
from matplotlib import pyplot as plt

print(stats.binom.pmf(k=3, n=5, p=0.5))

x = np.linspace(-4, 4, 200)
plt.plot(x, stats.norm.pdf(x, loc=0, scale=1))
plt.title('Normal Distribution')
plt.savefig('normal_distribution.png')
# plt.show()

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


np.random.seed(1)
group_a = np.random.binomial(1, 0.05, 200)
group_b = np.random.binomial(1, 0.07, 200)
t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"Group A conversion : {group_a.mean():.3f}")
print(f"Group B conversion : {group_b.mean():.3f}")
print(f"P-value : {p_value:.3f}")

data_1 = np.random.normal(50, 10, 30)
mean = np.mean(data_1)
sem = stats.sem(data_1) # standard error of the mean
ci = stats.t.interval(0.95, df=len(data_1)-1, loc=mean, scale=sem)
print(f"Mean : {mean:.3f}, 95% CI : {ci}")


heads, tails = 8, 2
p_values = np.linspace(0.01, 0.99, 100)
likelihood = (p_values ** heads) * ((1 - p_values) ** tails)

plt.plot(p_values, likelihood)
plt.xlabel('p (probability of heads)')
plt.ylabel('likelihood of observed data')
plt.axvline(p_values[np.argmax(likelihood)], linestyle='--', color='r', label='MLE estimate')
plt.legend()
plt.savefig('likelihood_distribution.png')
# plt.show()
print("MLE Estimation: ", p_values[np.argmax(likelihood)])

raw_data = np.random.exponential(scale=1.0, size=100_000)
plt.hist(raw_data, bins=50)
plt.title("Original skewed distribution")
plt.savefig('original_skewed_distribution.png')
plt.show()

sample_means = [np.mean(np.random.choice(raw_data, size=30)) for _ in range(1000)]
plt.hist(sample_means, bins=50)
plt.title("Distribution of sample means (Central Limit Theorem)")
plt.savefig('sample_means_distribution.png')
plt.show()