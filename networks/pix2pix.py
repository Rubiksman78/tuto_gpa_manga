import tensorflow as tf
from tensorflow import keras
from networks.neural_networks import Generator, Discriminator

class Pix2Pix(keras.Model):
    def __init__(self):
        super(Pix2Pix,self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def call(self,x):
        x = self.generator(x)
        x = self.discriminator(x)
        return x

    def compile(self,generator_optimizer,discriminator_optimizer,loss_object):
        super(Pix2Pix,self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.loss_object = loss_object

    def train_step(self,real_data,target_data):
        with tf.GradientTape() as disc_tape:
            fake_data = self.generator(real_data,training=True)
            real_output = self.discriminator(target_data,training=True)
            fake_output = self.discriminator(fake_data,training=True)
            disc_loss = self.discriminator_loss(real_output,fake_output)
        disc_grads = disc_tape.gradient(disc_loss,self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(disc_grads,self.discriminator.trainable_variables))
        with tf.GradientTape() as gen_tape:
            fake_data = self.generator(real_data,training=True)
            fake_output = self.discriminator(fake_data,training=True)
            gen_loss = self.generator_loss(fake_output)
        gen_grads = gen_tape.gradient(gen_loss,self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gen_grads,self.generator.trainable_variables))
        return {"gen_loss":gen_loss,"disc_loss":disc_loss}

    def discriminator_loss(self,real_output,fake_output):
        real_loss = self.loss_object(tf.ones_like(real_output),real_output)
        fake_loss = self.loss_object(tf.zeros_like(fake_output),fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self,fake_output):
        return self.loss_object(tf.ones_like(fake_output),fake_output)
        