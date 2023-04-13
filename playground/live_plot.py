import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

# Initialize the plot
fig, ax = plt.subplots()
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
(line,) = ax.plot([], [])

# Generate data points
x = np.linspace(0, 100, 100)
y = np.random.rand(100)

# Update the plot as data points are generated
for i in range(len(x)):
    line.set_data(x[:i], y[:i])
    plt.draw()
    plt.pause(0.01)

# Show the final plot
plt.show()
