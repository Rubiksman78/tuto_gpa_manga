import tensorflow as tf
from tensorflow import keras

class Pix2Pix(keras.model):
    def __init__(self):
        super(Pix2Pix,self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def call(self,x):
        x = self.generator(x)
        return x

    def train_step(self,real_data):
        return loss