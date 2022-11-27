import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,metrics

# Réseaux communs à CycleGAN et Pix2Pix

# Définition du générateur
class Generator(keras.Model):
    def __init__(self):
        super(Generator,self).__init__()
        self.model = keras.Sequential([
            layers.Conv2D(64,3,1,padding="same",input_shape=(256,256,1)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(128,3,2,padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(256,3,2,padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(512,3,2,padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(256,3,2,padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(128,3,2,padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(64,3,2,padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(3,3,1,padding="same"),
            layers.Activation("tanh")
               ])

    def call(self,x):
        x = self.model(x)
        return x

# Définition du discriminateur
class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.model = keras.Sequential([
            layers.Conv2D(64,3,2,padding="same",input_shape=(256,256,3)),
            layers.LeakyReLU(0.2),
            layers.Conv2D(128,3,2,padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(256,3,2,padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(512,3,2,padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(1,3,1,padding="same")
        ])

    def call(self,x):
        x = self.model(x)
        return x
