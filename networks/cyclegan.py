import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,metrics
from neural_networks import Generator,Discriminator

class CycleGAN(keras.model):
    def __init__(self):
        super(CycleGAN,self).__init__()
        self.generatorA2B = Generator()
        self.discriminatorA2B = Discriminator()
        self.generatorB2A = Generator()
        self.discriminatorB2A = Discriminator()

    def call(self,x,mode='A2B'):
        if mode == 'A2B':
            x = self.generatorA2B(x)
        elif mode == 'B2A':
            x = self.generatorB2A(x)
        return x

    def train_step(self,real_data):

        return loss
