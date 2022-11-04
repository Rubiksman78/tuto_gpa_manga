from networks.pix2pix import Pix2Pix
from process_data import load_dataset
from scripts.utils import *

dataset = load_dataset()
model = Pix2Pix()

def train(model,epochs):
    for epoch in range(epochs):
        for real_data in dataset:
            loss = model.train_step(real_data)
            print(loss)

