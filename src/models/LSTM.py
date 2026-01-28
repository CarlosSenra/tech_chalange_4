import tensorflow as tf
from tensorflow import keras
from keras import layers 


class SimpleLSTM(keras.Model):
    def __init__(self, units:int=50, output_dim:int=5):
        super(SimpleLSTM, self).__init__()

        self.lstm1 = layers.LSTM(150,return_sequences=True, name='lstm_layer1')
        self.lstm2 = layers.LSTM(units, name='lstm_layer2')
        self.lstm2 = layers.LSTM(units, name='lstm_layer2')
        self.dense = layers.Dense(output_dim, name='output_layer')

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        output = self.dense(x)

        return output