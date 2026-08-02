from scipy import stats
import numpy as np
from matplotlib import pyplot as plt

print(stats.binom.pmf(k=3, n=5, p=0.5))

x = np.linspace(-4, 4, 200)
plt.plot(x, stats.norm.pdf(x, loc=0, scale=1))
plt.title('Normal Distribution')
plt.savefig('normal_distribution.png')
plt.show()
