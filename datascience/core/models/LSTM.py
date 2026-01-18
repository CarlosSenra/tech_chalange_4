import tensorflow as tf
from tensorflow import keras
from keras import layers 


class SimpleLSTM(keras.Model):
    def __init__(self, units:int=50, output_dim:int=1):
        super(SimpleLSTM, self).__init__()

        self.lstm = layers.LSTM(units, name='lstm_layer')
        self.dense - layers.Dense(output_dim, name='output_layer')

    def call(self, inputs):
        x = self.lstm(inputs)
        output = self.dense(x)

        return output