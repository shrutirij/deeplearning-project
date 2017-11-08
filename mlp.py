'''
Multi-Layer Perceptron in DyNet.
Contains separate classes for Layers and the model to make it generic.
'''

import dynet as dy
import numpy as np

'''
Class definition for a layer in MLP with the weights and the bias parameters.
'''
class Layer(object):
    def __init__(self, input_dim, output_dim):
        '''
        input_dim: Input dimension of the layer
        output_dim: Output dimension of the layer
        '''
        self.input_dim = input_dim
        self.output_dim = output_dim


'''
Creates layers according to given dimensions and returns a list
'''
def create_layers(num_layers, layer_dims):
    layer_list = []
    for l in range(num_layers):
        temp_l = Layer(layer_dims[l][0], layer_dims[l][1])
        layer_list.append(temp_l)
    return layer_list

'''
Creates and returns dynet params for each layer object for the model.
Assumption - the input param has been initialized with proper dimensions.
Return: the updated model and the list of parameter objects
'''
def create_params(model, layers):
    param_list = []
    for l in range(len(layers)):
        W = model.add_parameters((layers[l].output_dim, layers[l].input_dim))
        b = model.add_parameters((layers[l].output_dim))
        param_list.append((W, b))
    return param_list

def get_activation(act_str):
    if act_str == "tanh":
        return dy.tanh
    elif act_str == "rectify" or act_str == "relu":
        return dy.rectify
    elif act_str == "sigmoid" or act_str == "logistic":
        return dy.logistic
    elif act_str == "softmax":
        return dy.softmax
    else:
        return dy.tanh


'''
Class Definition for MLP
'''
class MLP(object):
    def __init__(self, num_layers, layer_dims, activation, drop_rate):
        '''
        num_layers: number of layers in the network (including the output layer)
        layer_dims: List of layer dimension tuples [(i1,h1), (h1,h2), (h2,h3)...]
        activation: Activation function used for the layer (tanh, rectify, logistic)
        drop_rate: dropout rate if any
        '''
        self.num_layers = num_layers
        self.layer_dims = layer_dims
        self.activation = activation
        self.drop_rate = drop_rate
        self.model = dy.Model()
        self.layers = create_layers(num_layers, layer_dims)
        self.params = create_params(self.model, self.layers)

    def forward(self, input_expr):
        h_cur = input_expr
        # Feed forward through all but the last layer
        L = self.num_layers
        for l in range(L-1):
            #dy.renew_cg()
            w_cur = dy.parameter(self.params[l][0])
            b_cur = dy.parameter(self.params[l][1])
            f_act = get_activation(self.activation)
            h_new = f_act((w_cur * h_cur) + b_cur)
            h_cur = h_new

        # Return logistic sigmoid for binary classification for the last layer
        w_last = dy.parameter(self.params[L-1][0])
        b_last = dy.parameter(self.params[L-1][1])
        y_pred = dy.logistic((w_last * h_cur) + b_last)
        return y_pred










