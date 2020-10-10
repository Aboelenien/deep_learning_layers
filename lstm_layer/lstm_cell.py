import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import activations
from tensorflow.python.keras.engine.base_layer import Layer
import numpy as np

class LstmCell(Layer):

    def __init__(self, units, use_bias = True, **kwargs):
        super(LstmCell, self).__init__(**kwargs)
        self.activation = activations.get('tanh')
        self.recurrent_activation = activations.get('hard_sigmoid')
        self.units = units
        self.use_bias = use_bias
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape= (input_dim, self.units * 4),
            name = 'kernel')
        self.kernel_i = self.kernel[:, self.units]
        self.kernel_f = self.kernel[:, self.units:self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2:self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel = self.add_weight(shape=(input_dim, self.units * 4), name = "recurrent_kernel")
        self.recurrent_kernel_i = self.recurrent_kernel[:, self.units]
        self.recurrent_kernel_f = (self.recurrent_kernel[:, self.units: self.units * 2])
        self.recurrent_kernel_c = (self.recurrent_kernel[:, self.units * 2: self.units * 3])
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]
        
        if self.use_bias:
            print(self.units)
            self.bias = self.add_weight(shape=(self.units * 4, ), name = "bias") 
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_c = self.bias[self.units * 2: self.units * 3]
            self.bias_o = self.bias[self.units * 3:]
        else:
            self.bias = None
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None

        self.built = True

    def call(self, inputs, states):
        input_i = inputs
        input_f = inputs
        input_c = inputs
        input_o = inputs

        w_x_i = K.dot(input_i, self.kernel_i)
        w_x_f = K.dot(input_f, self.kernel_f)
        w_x_c = K.dot(input_c, self.kernel_c)
        w_x_o = K.dot(input_o, self.kernel_o)

        if use_bias:
            w_x_i = K.add_bias(w_x_i, self.bias_i)
            w_x_f = K.add_bias(w_x_f, self.bias_f)
            w_x_c = K.add_bias(w_x_c, self.bias_c)
            w_x_o = K.add_bias(w_x_o, self.bias_o)
    
        h_tm1 = states[0]
        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1

        i = self.recurrent_activation(w_x_i + K.dot(h_tm1_i, self.recurrent_kernel_i))
        f = self.recurrent_activation(w_x_f + K.dot(h_tm1_f, self.recurrent_kernel_f))
        o = self.recurrent_activation(w_x_o + K.dot(h_tm1_o, self.recurrent_kernel_o))
        c = self.activation(w_x_c + K.dot(h_tm1_c, self.recurrent_kernel_c))
        c = f * states[1]  + i * c
        h = o * self.recurrent_activation(c)
        return h, [h, c]
    


if __name__ == "__main__":
    lstm_cell = LstmCell(units = 32)
    lstm_cell.build((5, 20))