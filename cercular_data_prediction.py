import numpy as np

XOR_dataset = [
    (np.array([[0.0], [0.0]]), np.array([[0.0]])),
    (np.array([[0.0], [1.0]]), np.array([[1.0]])),
    (np.array([[1.0], [0.0]]), np.array([[1.0]])),
    (np.array([[1.0], [1.0]]), np.array([[0.0]])),
]


def initialize_network(layer_sizes):
    weights = []
    biases = []

    for i in range(len(layer_sizes) - 1):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i + 1]

        W = np.random.uniform(-1, 1, (output_size, input_size))
        b = np.random.uniform(-1, 1, (output_size, 1))

        weights.append(W)
        biases.append(b)

    return weights, biases


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def forward(weights, biases, x):
    activations = [x]
    zs = []

    a = x

    for i, (W, b) in enumerate(zip(weights, biases)):
        z = W @ a + b
        zs.append(z)
        #ReLu est meilleur pour les couches cachees, alors que sigmoid est meilleur pour les couches outputs 
        if i == len(weights) - 1:
            # couche de sortie : sigmoid 
            a = sigmoid(z)
        else:
            # couches cachées : ReLU
            a = relu(z)

        activations.append(a)

    return activations, zs


def backprop(x, y, weights, biases, lr):
    activations, zs = forward(weights, biases, x)

    deltas = [None] * len(weights)

    # delta de sortie : sigmoid + cross entropy
    deltas[-1] = activations[-1] - y

    # deltas cachés
    for l in range(len(weights) - 2, -1, -1):
        deltas[l] = (weights[l + 1].T @ deltas[l + 1]) * relu_derivative(zs[l])

    # mise à jour des poids et biais
    for l in range(len(weights)):
        weights[l] -= lr * (deltas[l] @ activations[l].T)
        biases[l] -= lr * deltas[l]

    return weights, biases


def train(dataset, weights, biases, lr, epochs):
    for epoch in range(epochs):
        total_loss = 0.0

        for x, y in dataset:
            activations, _ = forward(weights, biases, x)
            pred = activations[-1]

            eps = 1e-12
            pred_clipped = np.clip(pred, eps, 1 - eps)
            loss = -(y * np.log(pred_clipped) + (1 - y) * np.log(1 - pred_clipped))
            total_loss += float(loss.sum())

            weights, biases = backprop(x, y, weights, biases, lr)

        if epoch % 5000 == 0:
            print(f"epoch {epoch} loss {total_loss:.6f}")

    return weights, biases

def generate_circle_dataset(n_samples=2000, radius=0.5):
    dataset = []

    for _ in range(n_samples):

        x1 = np.random.uniform(-1, 1)
        x2 = np.random.uniform(-1, 1)

        label = 1 if x1**2 + x2**2 < radius**2 else 0

        x = np.array([[x1], [x2]])
        y = np.array([[label]])

        dataset.append((x, y))

    return dataset

if __name__ == "__main__":
    np.random.seed(42)

    train_data = generate_circle_dataset(2000)
    test_data = generate_circle_dataset(500)

    layer_sizes = [2, 16, 8, 1]

    weights, biases = initialize_network(layer_sizes)

    weights, biases = train(train_data, weights, biases, lr=0.1, epochs=200)

    correct = 0

    for x, y in test_data:

        activations, _ = forward(weights, biases, x)
        pred = activations[-1][0,0]

        predicted_class = 1 if pred >= 0.5 else 0

        if predicted_class == int(y[0,0]):
            correct += 1

    accuracy = correct / len(test_data)

    print("Accuracy:", accuracy)