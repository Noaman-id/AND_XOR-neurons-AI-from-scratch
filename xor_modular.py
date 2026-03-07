#Ces formules correspondent à la rétropropagation standard des réseaux feedforward, comme présentée dans 
#Goodfellow, Bengio, Courville — Deep Learning et Michael Nielsen — Neural Networks and Deep Learning.

import math
import random

XOR_dataset = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

def initialize_network(layer_sizes):
    weights = []
    biases = []

    for i in range(len(layer_sizes) - 1):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i + 1]

        layer_weights = []
        layer_biases = []

        for _ in range(output_size):
            neuron_weights = []
            for _ in range(input_size):
                neuron_weights.append(random.uniform(-1, 1))
            layer_weights.append(neuron_weights)

            layer_biases.append(random.uniform(-1, 1))

        weights.append(layer_weights)
        biases.append(layer_biases)

    return weights, biases

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def neuron_forward(weights, inputs, bias):
    z = 0
    for w, x in zip(weights, inputs):
        z += w * x
    z += bias
    a = sigmoid(z)
    return a

def layer_forward(layer_weights, inputs, layer_biases):
    outputs = []

    for weights, bias in zip(layer_weights, layer_biases):
        neuron_output = neuron_forward(weights, inputs, bias)
        outputs.append(neuron_output)

    return outputs

def full_forward(weights, biases, x):
    activations = [x]
    current_input = x

    for layer_weights, layer_biases in zip(weights, biases):
        current_output = layer_forward(layer_weights, current_input, layer_biases)
        activations.append(current_output)
        current_input = current_output

    return activations

def sigmoid_derivative(a):
    return a * (1 - a)

def compute_hidden_deltas(hidden_outputs, next_weights, next_deltas):
    hidden_deltas = []

    for j in range(len(hidden_outputs)):
        h = hidden_outputs[j]

        weighted_error = 0
        for k in range(len(next_deltas)):
            weighted_error += next_weights[k][j] * next_deltas[k]

        delta = weighted_error * sigmoid_derivative(h)
        hidden_deltas.append(delta)

    return hidden_deltas

def compute_output_deltas(pred, target):
    deltas = []
    for i in range(len(pred)):
        deltas.append(pred[i] - target[i])

    return deltas

def backprop(x, target, weights, biases, lr):
    target = [target]

    # ---------- forward ----------
    activations = full_forward(weights, biases, x)

    # deltas[0] correspond à l'entrée, donc inutile
    deltas = [None] * len(activations)

    # ---------- delta sortie ----------
    deltas[-1] = compute_output_deltas(activations[-1], target)

    # ---------- deltas cachés ----------
    for layer_index in range(len(activations) - 2, 0, -1):
        current_outputs = activations[layer_index]
        next_weights = weights[layer_index]
        next_deltas = deltas[layer_index + 1]

        deltas[layer_index] = compute_hidden_deltas(
            current_outputs,
            next_weights,
            next_deltas
        )

    # ---------- update weights and biases ----------
    for layer_index in range(len(weights) - 1, -1, -1):
        current_deltas = deltas[layer_index + 1]
        previous_activations = activations[layer_index]

        for i in range(len(weights[layer_index])):
            for j in range(len(previous_activations)):
                weights[layer_index][i][j] -= lr * current_deltas[i] * previous_activations[j]

            biases[layer_index][i] -= lr * current_deltas[i]

    return weights, biases

def train_network(dataset, weights, biases, lr, epochs):
    for epoch in range(epochs):
        for x, target in dataset:
            weights, biases = backprop(x, target, weights, biases, lr)

    return weights, biases

if __name__ == "__main__":

    layer_sizes = [2, 3, 1]
    weights, biases = initialize_network(layer_sizes)

    weights, biases = train_network(XOR_dataset, weights, biases, 0.1, 50000)

    print("\nTesting network\n")

    for x, target in XOR_dataset:
        activations = full_forward(weights, biases, x)
        pred = activations[-1][0]
        predicted_class = 1 if pred >= 0.5 else 0

        print(f"Input: {x}")
        print(f"Activations: {activations}")
        print(f"Prediction: {pred:.4f}")
        print(f"Predicted class: {predicted_class} | Target: {target}")
        print()