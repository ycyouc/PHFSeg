import matplotlib.pyplot as plt
import numpy as np

def prelu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

x = np.linspace(-5, 5, 1000)
y = prelu(x)

plt.plot(x, y)
plt.title('PReLU Function')
plt.xlabel('x')
plt.ylabel('f(x)')
ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.grid(True)
plt.show()
