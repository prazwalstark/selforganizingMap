import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv


# Define SOM parameters
neurons = 1000
epochs = 80
sigo = 500
no = 0.9


# Calculate constants for learning rate and neighborhood radius
r1 = epochs / np.log(no)
r2 = epochs / np.log(sigo)


# Read data from CSV
with open('dataset.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)


data = np.array(data, dtype=float)


# Extract x and y values
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)


# Normalize data to range [0, 1]
x_n = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
y_n = (y - np.amin(y)) / (np.amax(y) - np.amin(y))


# Initialize weights randomly
w = np.random.rand(neurons + 1, 2)


# Create figure and scatter plot
fig, ax = plt.subplots()
scatter = ax.scatter(x_n, y_n, s=10, color='blue') # Smaller node dots
path, = ax.plot([], [], color='red', linestyle='-')

# Initialize variables for shortest path calculation
shortest_path = None
shortest_distance = float('inf')


# Function to update the path in each animation frame
def update(frame):
    global w, shortest_path, shortest_distance
    for i in range(len(x)):
        x_input = np.array([x_n[i], y_n[i]]).reshape(1, 2)
        xw = np.zeros((neurons, 1))
        for j in range(neurons):
            xw[j] = ((x_input[0, 0] - w[j, 0]) * (x_input[0, 0] - w[j, 0]) + (x_input[0, 1] - w[j, 1]) * (x_input[0, 1] - w[j, 1]))
        i_x = np.argmin(xw)
        d = np.zeros((neurons, 1))
        for k in range(neurons):
            d[k] = np.minimum(np.abs(i_x - k), neurons - np.abs(i_x - k))
        # weight update
        n = no * np.exp(-1 * frame / r1)
        sig = sigo * np.exp(-1 * frame / r2)


        for l in range(neurons):
            w[l, :] = w[l, :] + n * np.exp(-1 * d[l] * d[l] / (2 * sig * sig)) * (x_input - w[l, :])


    # Close the loop by connecting the last neuron to the first neuron
    w[neurons, 0] = w[0, 0]
    w[neurons, 1] = w[0, 1]


    # Update path (moving line)
    path.set_data(w[:, 0], w[:, 1])

    # Calculate shortest path and distance
    current_distance = np.sum(np.linalg.norm(w[1:] - w[:-1], axis=1))
    if current_distance < shortest_distance:
        shortest_distance = current_distance
        shortest_path = np.copy(w[:-1, :])

    # Stop the animation when the path is completed
    if frame == epochs - 1:
        ani.event_source.stop()

    # Print the current shortest path and distance
    print(f"Shortest Path: {shortest_path}")
    print(f"Shortest Distance: {shortest_distance}")

    return path,


# Set plot limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)


# Create animation
ani = FuncAnimation(fig, update, frames=epochs, interval=0.1, repeat=False)


plt.title("Travelling Salesman Problem")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.show()
