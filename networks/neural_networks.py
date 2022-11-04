import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,metrics

# Réseaux communs à CycleGAN et Pix2Pix

# Définition du générateur
class Generator(keras.model):
    def __init__(self):
        super(Generator,self).__init__()
        self.model = self.Sequential([
        ])

    def call(self,x):
        x = self.model(x)
        return x

# Définition du discriminateur
class Discriminator(keras.model):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.model = self.Sequential([
        ])

    def call(self,x):
        x = self.model(x)
        return x