import random
import math

OR_dataset = [
    ((0,0), 0),
    ((0,1), 1),
    ((1,0), 1),
    ((1,1), 1),
]

def forward(weights, inputs, bias)->float:
    
    dotproduct = 0
    for weight, input in zip(weights, inputs):
        dotproduct += weight*input
    z = dotproduct + bias

    pred = 1 / (1 + math.exp(-z))
    return pred

def loss(pred, target):
    return -( target*math.log(pred) + (1-target)*math.log(1-pred ))

def gradiant(pred, target):
    return pred - target

def parameter_update(weights, bias, lr, gredient, inputs):
    new_weights = []
    for i in range(len(weights)):
        new_weight =  weights[i] - lr * gredient * inputs[i]
        new_weights.append(new_weight)

    bias = bias - lr * gredient
    return new_weights, bias

def train(epochs,weights,bias,lr):
    
    for epoch in range(epochs):
        total_loss = 0
        for input,target in OR_dataset:    
            pred = forward(weights,input,bias)
            l = loss(pred,target)
            total_loss += l
            grad= gradiant(pred, target)
            weights,bias = parameter_update(weights,bias,lr,grad,input)

            if epoch % 100 == 0:
                print(epoch, total_loss)

    return weights, bias

def test(weights, bias):
    print("\nTesting model")

    for inputs, target in OR_dataset:
        pred = forward(weights, inputs, bias)

        predicted_class = 1 if pred >= 0.5 else 0

        print(
            f"Input: {inputs} | "
            f"Predicted probability: {pred:.4f} | "
            f"Predicted class: {predicted_class} | "
            f"Target: {target}"
        )

if __name__ == "__main__":
    weights = [random.uniform(-1,1),random.uniform(-1,1)]
    bias = random.uniform(-1,1)

    weights,bias = train(3000,weights,bias,0.1)

    test(weights,bias)