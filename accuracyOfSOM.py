import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class MiniSom:
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5):
        self.weights = np.random.rand(x, y, input_len)
        self.sigma = sigma
        self.learning_rate = learning_rate

    def update_weights(self, x, winner, lr):
        delta_weights = np.zeros_like(self.weights)
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                distance = np.linalg.norm([i - winner[0], j - winner[1]])
                influence = np.exp(-distance**2 / (2 * self.sigma**2))
                delta_weights[i, j] = lr * influence * (x - self.weights[i, j])
        self.weights += delta_weights

    def find_bmu(self, x):
        distances = np.linalg.norm(x.reshape(1, -1) - self.weights.reshape(-1, self.weights.shape[-1]), axis=1)
        return np.unravel_index(np.argmin(distances), self.weights.shape[:2])

    def train_random(self, data, epochs):
        for epoch in range(epochs):
            learning_rate = self.learning_rate / (1 + epoch)
            np.random.shuffle(data)
            for x in data:
                bmu = self.find_bmu(x)
                self.update_weights(x, bmu, learning_rate)

    def predict(self, data):
        return np.array([self.find_bmu(x) for x in data])

# Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Normalize data to range [0, 1]
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Define SOM parameters
grid_size = (10, 10)  # no of neurons
epochs = 100  # Number of epochs
initial_learning_rate = 0.1

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

som = MiniSom(grid_size[0], grid_size[1], X_train.shape[1])
som.train_random(X_train, epochs)

train_predicted_bmus = som.predict(X_train)
predicted_bmus = som.predict(X_test)

predicted_labels = np.array([y_train[predicted_bmu[0]] for predicted_bmu in predicted_bmus])
train_predicted_labels = np.array([y_train[predicted_bmu[0]] for predicted_bmu in train_predicted_bmus])


# Calculate accuracy
accuracy = accuracy_score(y_test, predicted_labels)
print("TestSet-Accuracy of the Self-Organizing Map for Wine dataset:", accuracy)

train_accuracy=accuracy_score(y_train,train_predicted_labels )
print("TrainSet-Accuracy of the Self-Organizing Map for Wine dataset:",train_accuracy)
