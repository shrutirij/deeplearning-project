import dynet as dy
import numpy as np
import mlp

layer_dims = [(10,100), (100,1)]
num_layers = 2
m = mlp.MLP(num_layers, layer_dims, "tanh", 0.0)
input_x = np.random.uniform(0, 1, (10))
x1 = dy.vecInput(10)
x1.set(input_x)
pred = m.forward(x1)
print pred.value()