# -*- coding: utf-8 -*-
"""
# Implementation of a CycleGAN Model for MRI Image Translation
"""

# pip install nibabel pydicom medpy
# pip install git+https://www.github.com/keras-team/keras-contrib.git
# pip install visualkeras

# Import Packages
import os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Activation
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Dropout
from keras.initializers import RandomNormal
from keras.layers import Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

from medpy.io import load
import pylab as plt
import nibabel as nb
import numpy as np
import glob

import scipy.misc
import numpy.random as rng
from PIL import Image, ImageDraw, ImageFont
from sklearn.utils import shuffle
import nibabel as nib #reading MR images
import math
from matplotlib import pyplot as plt
import visualkeras

# Load Data

ff = glob.glob('./data/MRI/*.mha')

len(ff)

images = []

from skimage.transform import resize

for f in range(len(ff)):
    a, a_header = load(ff[f])
    print(a.shape)
    a = a[:,:,200:285]
    a = resize(a, (256,256))
    for i in range(a.shape[2]):
        images.append((a[:,:,i]))
print (a.shape)

print(a.shape)

# Data Preprocessing

images = np.asarray(images)

images.shape

images = images.reshape(-1,256,256,1)

images.shape

m = np.max(images)
mi = np.min(images)

images = (images - mi) / (m - mi)

np.min(images), np.max(images)

# Split Data - Dataset A, 3.0T

from sklearn.model_selection import train_test_split
trainA, testA, train_ground, test_ground = train_test_split(images, images, test_size=0.3, random_state=1)

# Data Exploration

print("Dataset (images) shape: {shape}".format(shape=images.shape))

plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(trainA[1], (256,256))
plt.imshow(curr_img, cmap='gray')

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(testA[1], (256,256))
plt.imshow(curr_img, cmap='gray')

# DCGAN Model

# Standalone Discriminator Model
def gan_discriminator(in_shape=(256,256,1)):
  model = Sequential()

  model.add(Conv2D(32, (4,4), strides = (2, 2), padding = "same", input_shape = in_shape))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.4))

  model.add(Conv2D(64, (4,4), strides = (2, 2), padding = "same"))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.4))

  model.add(Conv2D(128, (4,4), strides = (2,2), padding = "same"))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.4))

  model.add(Flatten())
  model.add(Dense(1, activation = "sigmoid"))

  # Compile
  opt = Adam(learning_rate=0.0002, beta_1=0.5)
  model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])
  return model

# Define model
model = gan_discriminator()

# Summarize model
model.summary()

# Plot Model
# plot_model(model, to_file= "discriminator_plot.png", show_shapes=True, show_layer_names=True)

# Standalone Generator Model
def gan_generator(latent_dim):
  model = Sequential()

  # foundation for 32x32 image
  n_nodes = 256 * 32 * 32
  model.add(Dense(n_nodes, input_dim = latent_dim))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Reshape((32, 32, 256)))

  # upsample to 64x64
  model.add(Conv2DTranspose(256, (4,4), strides = (2,2), padding = "same"))
  model.add(LeakyReLU(alpha=0.2))

  # upsample to 128x128
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding = "same"))
  model.add(LeakyReLU(alpha=0.2))

  # upsample to 256x256
  model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding = "same"))
  model.add(LeakyReLU(alpha=0.2))

  model.add(Conv2D(1, (3,3), activation = "sigmoid", padding = "same"))
  return model


# define the size of the latent space
latent_dim = 100
# define the generator model
model = gan_generator(latent_dim)
# summarize the model
model.summary()
# plot the model
# plot_model(model, to_file="generator_plot.png", show_shapes=True, show_layer_names=True)

# Combined GAN Model
def define_gan(g_model, d_model):
  # make weights in the discriminator not trainable
  d_model.trainable = False

  # connect them
  model = Sequential()

  # add generator
  model.add(g_model)

  # add the discriminator
  model.add(d_model)

  # compile model
  opt = Adam(learning_rate = 0.0002, beta_1 = 0.5)
  model.compile(loss = "binary_crossentropy", optimizer = opt)
  return model

# size of the latent space
latent_dim = 100
# create the discriminator
d_model = gan_discriminator()
# create the generator
g_model = gan_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# summarize gan model
gan_model.summary()
# plot gan model
# plot_model(gan_model, to_file="gan_plot.png", show_shapes=True, show_layer_names=True)

# Select real samples
def generate_real_samples(dataset, n_samples):
  # Choose random instances
  ix = randint(0, dataset.shape[0], n_samples)

  # Retrieve selected images
  X = dataset[ix]

  # Generate real class labels (1)
  y = ones((n_samples, 1))
  return X, y

# Generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
  # Generate points in the latent space
  x_input = randn(latent_dim * n_samples)

  # Reshape into a batch of inputs for the network
  x_input = x_input.reshape(n_samples, latent_dim)
  return x_input

# Generate fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
  # generate points in latent space
  x_input = generate_latent_points(latent_dim, n_samples)

  # predict outputs
  X = g_model.predict(x_input)

  # create fake class labels (0)
  y = zeros((n_samples, 1))
  return X, y

from matplotlib import pyplot
# create and save a plot of generated images
def save_plot(examples, epoch, n=10):
  # plot images
  for i in range(n * n):
    # define subplot
    pyplot.subplot(n, n, 1 + i)
    # turn off axis
    pyplot.axis('off')
    # plot raw pixel data
    pyplot.imshow(examples[i], cmap='gray')
    # pyplot.imshow(examples[i, :, :, 0], cmap='gray')
    # save plot to file
    filename = './figures/DCGAN/generated3T_plot_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()

# Evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
  # prepare real samples
  X_real, y_real = generate_real_samples(dataset, n_samples)

  # evaluate discriminator on real examples
  _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)

  # prepare fake examples
  x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)

  # evaluate discriminator on fake examples
  _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)

  # summarize discriminator performance
  print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))

  # save plot
  save_plot(x_fake, epoch)

  # save the generator model tile file
  filename = './models/DCGAN/generator3T_model_%03d.h5' % (epoch + 1)
  g_model.save(filename)

# Train the composite model
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=60, n_batch=10):
  bat_per_epo = int(dataset.shape[0] / n_batch)
  half_batch = int(n_batch / 2)
  # manually enumerate epochs
  for i in range(n_epochs):
    # enumerate batches over the training set
    for j in range(bat_per_epo):
      # get randomly selected real samples
      X_real, y_real = generate_real_samples(dataset, half_batch)
      # generate fake examples
      X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
      # create training set for the discriminator
      X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
      # update discriminator model weights
      d_loss, _ = d_model.train_on_batch(X, y)
      # prepare points in latent space as input for the generator
      X_gan = generate_latent_points(latent_dim, n_batch)
      # create inverted labels for the fake samples
      y_gan = ones((n_batch, 1))
      # update the generator via the discriminator's error
      g_loss = gan_model.train_on_batch(X_gan, y_gan)
      # summarize loss on this batch
      print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
    # evaluate the model performance, sometimes
    if (i+1) % 10 == 0:
      summarize_performance(i, g_model, d_model, dataset, latent_dim)

# size of the latent space
latent_dim = 100
# create the discriminator
d_model = gan_discriminator()
# create the generator
g_model = gan_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = trainA
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)

# Use Final DCGAN Generator

# example of loading the generator model and generating images
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
  # generate points in the latent space
  x_input = randn(latent_dim * n_samples)
  # reshape into a batch of inputs for the network
  x_input = x_input.reshape(n_samples, latent_dim)
  return x_input

# create and save a plot of generated images
def save_plot(examples, n):
  # plot images
  for i in range(n * n):
    # define subplot
    pyplot.subplot(n, n, 1 + i)
    # turn off axis
    pyplot.axis('off')
    # plot raw pixel data
    pyplot.imshow(examples[i,:,:], cmap='gray')
    # pyplot.imshow(examples[i, :, :, 0], cmap='gray')
  pyplot.show()

# load model
model = load_model('./models/DCGAN/generator3T_model_050.h5')
# generate images
latent_points = generate_latent_points(100, 9)
# generate images
X = model.predict(latent_points)
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# plot the result
save_plot(X, 3)