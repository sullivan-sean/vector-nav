import numpy as np
import matplotlib.pyplot as plt


losses = [
    7.87524,
    7.74761,
    7.70878,
    7.66657,
    7.62009,
    7.51256,
    7.35644,
    7.24392,
    7.13170,
    7.01272,
    6.85325,
    6.71447,
    6.60848,
    6.52398,
    6.45931,
    6.40425,
    6.34920,
    6.30833,
    6.26645,
    6.22407,
]

losses_2 = [
    7.642731852555275,
    7.178092469000816,
    7.00948871538639,
    6.919548381495476,
]

plt.plot(np.arange(len(losses)), losses)
plt.plot(np.arange(len(losses_2)), losses_2)
plt.show()
