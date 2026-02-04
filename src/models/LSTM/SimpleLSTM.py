import tensorflow as tf
from tensorflow import keras
from keras import layers 


class SimpleLSTM(keras.Model):
    def __init__(self, list_units:list[int], output_dim:int):
        self.list_units = list_units
        self.output_dim = output_dim
        super(SimpleLSTM, self).__init__()
        for i, units in enumerate(self.list_units):
            return_seq = i < len(self.list_units) - 1
            setattr(self, f'lstm_{i+1}', layers.LSTM(units, return_sequences=return_seq, name=f'lstm_layer_{i+1}'))

        self.dense = layers.Dense(self.output_dim, name='output_layer')

    def call(self, inputs):
        x = inputs
        for i in range(len(self.list_units)):
            x = getattr(self, f'lstm_{i+1}')(x)
        y = self.dense(x)
        return y