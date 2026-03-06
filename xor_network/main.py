import math

AND_dataset = [((1,1),1),
               ((0,1),0),
               ((1,0),0),
               ((0,0),0)]

XOR_dataset = [
    ((0,0), 0),
    ((0,1), 1),
    ((1,0), 1),
    ((1,1), 0),
]

def dot_product(input:list, weight:list):
    output_score = input[0] * weight[0] + input[1] * weight[1]
    return output_score

def bais_add(score, bias):
    output = score + bias
    return output

def activation_sigmoid(confidence):
    return 1/ (1 + math.exp(-confidence))

def prediction(inputs, weights, bias) :
    score = dot_product(inputs, weights)
    z = bais_add(score, bias)
    return activation_sigmoid(z)

def Loss(pred, target):
    return math.pow(pred - target,2)

def derivative_sigma(sigma):
    return sigma*(1 - sigma)

def derivative_Loss(pred, target):
    return 2*(pred - target)

def derivative_z(inputs):
    dz_dw1 = inputs[0]
    dz_dw2 = inputs[1]
    dz_db = 1
    return dz_dw1, dz_dw2, dz_db

def gradients(inputs, target, weights, bias):
    a = prediction(inputs, weights, bias)

    dL_da = derivative_Loss(a, target)
    da_dz = derivative_sigma(a)

    dz_dw1, dz_dw2, dz_db = derivative_z(inputs)

    dw1 = dz_dw1 * da_dz * dL_da
    dw2 = dz_dw2 * da_dz * dL_da
    db = dz_db * da_dz * dL_da

    return dw1, dw2, db

def Update_Weight(weights, bias, dw1, dw2, db ,learning_rate):
    weights[0] = weights[0] - learning_rate * dw1
    weights[1] = weights[1] - learning_rate * dw2
    bias =  bias - learning_rate * db

    return weights, bias

def hidden_forward(inputs, W_hidden, B_hidden):
    hidden_neuron1_score = bais_add( dot_product(inputs, W_hidden[0]), B_hidden[0])
    hidden_neuron2_score = bais_add( dot_product(inputs, W_hidden[1]), B_hidden[1])

    neuron1_activation = activation_sigmoid(hidden_neuron1_score)
    neuron2_activation = activation_sigmoid(hidden_neuron2_score)
    
    return neuron1_activation, neuron2_activation

def network_forward(inputs, W_hidden, B_hidden, W_out, b_out):
    h1, h2 = hidden_forward(inputs, W_hidden, B_hidden)
    output_neuron_score = bais_add( dot_product([h1, h2] , W_out), b_out)
    output_neuron_activation = activation_sigmoid(output_neuron_score)

    return output_neuron_activation

def output_delta(pred, target):
    DLoss = derivative_Loss(pred, target)
    DSigma = derivative_sigma(pred)
    return DLoss * DSigma 

def hidden_delta(delta_out, weight_out, hidden_activation):
    return delta_out * weight_out * derivative_sigma(hidden_activation)

def network_gredients(inputs, target, W_hidde, B_hidden, b_out, w_out):
    x1,x2 = inputs

    z1= bais_add( dot_product(inputs, W_hidde[0]),B_hidden[0])
    z2= bais_add( dot_product(inputs, W_hidde[1]),B_hidden[1])

    h1=activation_sigmoid(z1)
    h2=activation_sigmoid(z2)

    z_out = bais_add( dot_product([h1,h2], w_out), b_out)
    y_pred = activation_sigmoid(z_out)

    delta_out = output_delta(y_pred, target)
    delta_h1 = hidden_delta(delta_out, w_out[0],h1)
    delta_h2 = hidden_delta(delta_out, w_out[1],h2)

    dv1 = delta_out * h1
    dv2 = delta_out * h2 
    db_out = delta_out

    dw11 = delta_h1 * x1
    dw12 = delta_h1 * x2
    db1  = delta_h1

    dw21 = delta_h2 * x1
    dw22 = delta_h2 * x2
    db2  = delta_h2

    grads_hidden = [[dw11,dw12],[dw21,dw22]]
    grads_b_hidden = [db1,db2]
    grads_out = [dv1, dv2]

    return grads_hidden , grads_b_hidden, grads_out, db_out, y_pred

def update_network_params_using_Update_Weight(
    W_hidden, B_hidden, W_out, b_out,
    grads_hidden, grads_b_hidden, grads_out, db_out,
    lr
):
    W_hidden[0], B_hidden[0] = Update_Weight(
        W_hidden[0],
        B_hidden[0],
        grads_hidden[0][0],
        grads_hidden[0][1],
        grads_b_hidden[0],
        lr
    )

    W_hidden[1], B_hidden[1] = Update_Weight(
        W_hidden[1],
        B_hidden[1],
        grads_hidden[1][0],
        grads_hidden[1][1],
        grads_b_hidden[1],
        lr
    )

    W_out, b_out = Update_Weight(
        W_out,
        b_out,
        grads_out[0],
        grads_out[1],
        db_out,
        lr
    )

    return W_hidden, B_hidden, W_out, b_out

def train_network(dataset, W_hidden, B_hidden, W_out, b_out, lr, epoches):
    for epoch in range(epoches):
        total_loss = 0.0

        for inputs, target in dataset:
            grads_hidden, grads_b_hidden, grads_out, db_out, y_pred = network_gredients(
                inputs, target, W_hidden, B_hidden, b_out, W_out
            )

            total_loss += Loss(y_pred, target)

            W_hidden, B_hidden, W_out, b_out = update_network_params_using_Update_Weight(
                W_hidden, B_hidden, W_out, b_out,
                grads_hidden, grads_b_hidden, grads_out, db_out,
                lr
            )

        if epoch % 1000 == 0:
            print("epoch:", epoch, "loss:", total_loss)

    return W_hidden, B_hidden, W_out, b_out
    
def classify(a, threshold = 0.5):
    return 1 if a >= threshold else 0

def test_network(dataset, W_hidden, B_hidden, W_out, b_out):
    for inputs, target in dataset:
        a = network_forward(inputs, W_hidden, B_hidden, W_out, b_out)
        y_hat = classify(a)
        print(inputs, "pred=", round(a, 4), "class=", y_hat, "target=", target)

if __name__ == "__main__":
    import random

    W_hidden = [
        [random.uniform(-1, 1), random.uniform(-1, 1)],
        [random.uniform(-1, 1), random.uniform(-1, 1)]
    ]
    B_hidden = [random.uniform(-1, 1), random.uniform(-1, 1)]

    W_out = [random.uniform(-1, 1), random.uniform(-1, 1)]
    b_out = random.uniform(-1, 1)

    print("Before training:")
    test_network(XOR_dataset, W_hidden, B_hidden, W_out, b_out)

    lr = 0.5
    epochs = 20000

    W_hidden, B_hidden, W_out, b_out = train_network(
        XOR_dataset, W_hidden, B_hidden, W_out, b_out, lr, epochs
    )

    print("\nAfter training:")
    test_network(XOR_dataset, W_hidden, B_hidden, W_out, b_out)