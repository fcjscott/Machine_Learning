import matplotlib.pyplot as plt
import numpy as np

t = np.arange(-5., 5., 0.1)
plt.plot(t, (t**2 - 5))
plt.scatter(-3, 0, c='g')
plt.scatter(-1, 0, c='r')
plt.scatter(1, 0, c='g')
plt.scatter(3, 0, c='r')
plt.show()