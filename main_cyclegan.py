from networks.cyclegan import CycleGAN
from scripts.process_data import load_dataset
from scripts.utils import *

dataset = load_dataset()
model = CycleGAN()

def train(model,epochs):
    for epoch in range(epochs):
        for real_data in dataset:
            loss = model.train_step(real_data)
            print(loss)

