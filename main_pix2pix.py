from networks.pix2pix import Pix2Pix
from scripts.process_data import load_dataset,show_dataset
from scripts.utils import *
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import matplotlib.pyplot as plt

dataset = load_dataset()
show_dataset(dataset)
#Subset of the dataset
model = Pix2Pix()
model.build(input_shape=(None,256,256,1))
gen_opt = keras.optimizers.Adam(learning_rate=0.0002,beta_1=0.5)
disc_opt = keras.optimizers.Adam(learning_rate=0.0002,beta_1=0.5)
model.compile(gen_opt,disc_opt,keras.losses.BinaryCrossentropy(from_logits=True))
model.generator.summary()

def plot_images(model,dataset,n_images=5):
    fig,axes = plt.subplots(n_images,2,figsize=(10,10))
    for i,(real_data,target_data) in enumerate(dataset.take(n_images)):
        pred = model.generator(real_data,training=False)
        axes[i,0].imshow(real_data[0,...]*0.5+0.5)
        axes[i,1].imshow(pred[0,...]*0.5+0.5)
    plt.show()

def train(model,epochs):
    for epoch in range(epochs):
        progress_bar = tqdm(dataset)
        for real_data,target_data in progress_bar:
            loss_dict = model.train_step(real_data,target_data)
            gen_loss = loss_dict["gen_loss"]
            disc_loss = loss_dict["disc_loss"]
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs} | Gen Loss: {gen_loss} | Disc Loss: {disc_loss}")
        plot_images(model,dataset)

train(model,epochs=10)

