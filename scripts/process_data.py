import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

PATH_TO_DATA = "dataset"
IMG_WIDTH = 256
IMG_HEIGHT = 256
NUM_OF_DATA = 100
BATCH_SIZE = 1

def read_input_and_real_image(index: int):
    path_a = "A"
    path_b = "B"
    train_a_path = os.path.join(PATH_TO_DATA, path_a)
    train_b_path = os.path.join(PATH_TO_DATA, path_b)
    train_a_list = os.listdir(train_a_path)
    train_b_list= os.listdir(train_b_path)
    train_a = tf.io.read_file(os.path.join(train_a_path, train_a_list[index]))
    train_b = tf.io.read_file(os.path.join(train_b_path, train_b_list[index]))
    train_a = tf.image.decode_jpeg(train_a, channels=3)
    train_b = tf.image.decode_jpeg(train_b, channels=3)
    return train_a, train_b

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(
        input_image, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    real_image = tf.image.resize(
        real_image, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    return input_image, real_image

def normalize(input_image, real_image):
    input_image = tf.cast(input_image, dtype=tf.float32) 
    real_image = tf.cast(real_image, dtype=tf.float32) 
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image

def load_image_train(input_image, real_image):
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image

def data_generator():   
    max_number = NUM_OF_DATA
    counter = 1
    while True:
        input_image, real_image = read_input_and_real_image(counter)
        input_image, real_image = load_image_train(input_image, real_image)
        yield input_image, real_image
        
        counter += 1
        if counter > max_number:
            counter = 1

def load_dataset():
    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(),
        output_signature=(
            tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32))
    )

    train_dataset = train_dataset.shuffle(NUM_OF_DATA)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    return train_dataset

def show_dataset(dataset):
    for input_image, real_image in dataset.take(1):
        input_image = input_image.numpy()
        real_image = real_image.numpy()
        plt.figure(figsize=(10, 10))
        display_list = [input_image[0], real_image[0]]
        title = ['Input Image', 'Real Image']
        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
            plt.imshow(display_list[i]*0.5 + 0.5)
            plt.axis('off')
        plt.show()