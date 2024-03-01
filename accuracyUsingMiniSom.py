import numpy as np
from minisom import MiniSom
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

wine = load_wine()
data = wine.data
target = wine.target

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

grid_size = (10, 10)  # No of neurons

som = MiniSom(grid_size[0], grid_size[1], X_train.shape[1], sigma=1.0, learning_rate=0.1)

som.train_random(X_train, 1000)  # Epochs

winner_coordinates = np.array([som.winner(x) for x in X_train])
cluster_indices = np.ravel_multi_index(winner_coordinates.T, grid_size)

cluster_labels = {}
for i, (x, y) in enumerate(winner_coordinates):
    if cluster_indices[i] not in cluster_labels:
        cluster_labels[cluster_indices[i]] = []
    cluster_labels[cluster_indices[i]].append(y_train[i])

final_cluster_labels = {}
for key, value in cluster_labels.items():
    final_cluster_labels[key] = max(set(value), key=value.count)

train_predicted_labels = []
for x in X_train:
    bmu = som.winner(x)
    cluster_index = bmu[0] * grid_size[1] + bmu[1]
    train_predicted_labels.append(final_cluster_labels[cluster_index])

train_accuracy = accuracy_score(y_train, train_predicted_labels)
print("Training Set Accuracy:", train_accuracy)
predicted_labels = []
for x in X_test:
    bmu = som.winner(x)
    cluster_index = bmu[0] * grid_size[1] + bmu[1]
    predicted_labels.append(final_cluster_labels[cluster_index])

accuracy = accuracy_score(y_test, predicted_labels)
print("TestSetAccuracy:", accuracy)