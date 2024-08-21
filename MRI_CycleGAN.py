#!/usr/bin/env python
# coding: utf-8

# Implementation of a CycleGAN Model for MRI Image Translation

# pip install medpy
# pip install SimpleITK
# pip install scipy
# pip install -q tensorflow
# pip install -q h5py
# pip install git+https://www.github.com/keras-team/keras-contrib.git
# pip install -q visualkeras
# pip install nibabel
# pip install -U pydicom
# pip install scikit-image

# Import Packages
import os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import random
import h5py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
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

from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint

from medpy.io import load
import nibabel as nib #reading MR images

import pylab as plt
import numpy as np
import glob

import scipy.misc
import numpy.random as rng
from PIL import Image, ImageDraw, ImageFont
from sklearn.utils import shuffle
import math
from matplotlib import pyplot as plt
import visualkeras


# Load Data
ff = glob.glob('./data/MRI/*.mha')
ff1 = glob.glob('./data/MRI/*.nii.gz')


len(ff)
len(ff1)


ff[0]
ff1[0]


images = []
images1 = []


# Reshape 3T images
from skimage.transform import resize

for f in range(len(ff)):
    a, a_header = load(ff[f])
    a = a[:,:,155:165]
    a = resize(a, (256,256))
    for i in range(a.shape[2]):
        images.append((a[:,:,i]))
print (a.shape)

# for f in range(len(ff)):
#     a, a_header = load(ff[f])
# #     a = np.asarray(a.dataobj)
#     for i in range (a.shape[2]):
#         slice_data = a[:,:,i]
#         slice_data = resize(slice_data, (256,256))
#         images.append(slice_data)


# Reshape 1.5T images
from skimage.transform import resize

for f in range(len(ff1)):
    b = nib.load(ff1[f])
    b = b.dataobj[:,:,20:30,0]
    b = resize(b, (256,256))
    for i in range(b.shape[2]):
        images1.append((b[:,:,i]))
print (b.shape)

# for f in range(len(ff1)):
#     b = nib.load(ff1[f])
#     data = np.asarray(b.dataobj)
#     for i in range(data.shape[2]):
#         slice_data = data[:,:,i,0]
#         slice_data = resize(slice_data, (256,256))
#         images1.append(slice_data)


# Data Preprocessing
images = np.asarray(images)
images1 = np.asarray(images1)


images.shape
images1.shape


images = images.reshape(-1,256,256,1)
images1 = images1.reshape(-1,256,256,1)


images.shape
images1.shape


# Normalize images
m = np.max(images)
mi = np.min(images)

m1 = np.max(images1)
mi1 = np.min(images1)


images = (images - mi) / (m - mi)
images1 = (images1 - mi1) / (m1 - mi1)


# In[ ]:
np.min(images), np.max(images)


# In[ ]:
np.min(images1), np.max(images1)


# Split Data - Dataset A, 3.0T
from sklearn.model_selection import train_test_split
trainA, testA, train_ground, test_ground = train_test_split(images, images, test_size=0.3, random_state=1)


# In[ ]:
trainA.shape


# Split Data - Dataset B, 1.5T

# In[ ]:
from sklearn.model_selection import train_test_split
trainB, testB, trainB_ground, testB_ground = train_test_split(images1, images1, test_size=0.3, random_state=2)


# In[ ]:
trainB.shape


# Data Exploration
print("Dataset (images) shape: {shape}".format(shape=images.shape))
print("Dataset (images) shape: {shape}".format(shape=images1.shape))


plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(trainA[2], (256,256))
plt.imshow(curr_img, cmap='gray')

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(testA[2], (256,256))
plt.imshow(curr_img, cmap='gray')


# In[ ]:


plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(trainB[1], (256,256))
plt.imshow(curr_img, cmap='gray')

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(testB[1], (256,256))
plt.imshow(curr_img, cmap='gray')


# CycleGAN Model

# Discriminator Model
def discriminator(input_shape):
    # Initialize weights
    init = RandomNormal(stddev = 0.02)

    # Source image
    input_image = Input(shape = input_shape)

    # 64 filters
    d = Conv2D(64, (4,4), strides = (2,2), padding = "same", kernel_initializer = init)(input_image)
    d = LeakyReLU(alpha = 0.2)(d)

    # 128 filters
    d = Conv2D(128, (4,4), strides = (2,2), padding = "same", kernel_initializer = init)(d)
    d = InstanceNormalization(axis = -1)(d)  # Features normalized per feature map
    d = LeakyReLU(alpha = 0.2)(d)

    # 256 filters
    d = Conv2D(256, (4,4), strides = (2,2), padding = "same", kernel_initializer = init)(d)
    d = InstanceNormalization(axis = -1)(d)
    d = LeakyReLU(alpha = 0.2)(d)

    # 512 filters
    d = Conv2D(512, (4,4), strides = (2,2), padding = "same", kernel_initializer = init)(d)
    d = InstanceNormalization(axis = -1)(d)
    d = LeakyReLU(alpha = 0.2)(d)

    #1 filter
    d = Conv2D(512, (4,4), strides = (2,2), padding = "same", kernel_initializer = init)(d)
    d = InstanceNormalization(axis = -1)(d)
    d = LeakyReLU(alpha = 0.2)(d)

    # Patch output
    patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)

    # Define model
    model1 = Model(input_image, patch_out)
    
    # Optimization algorithm
    opt = Adam(learning_rate = 0.0002, beta_1 = 0.5)

    # Compile model
    model1.compile(loss = "mse", optimizer = opt, loss_weights = [0.5])
    return model1


# Define image shape
input_shape = (256, 256, 1)

# Create model
model1 = discriminator(input_shape)

# Summarize model
model1.summary()

# Plot
# plot_model(model1, to_file= "discriminator_model.png", show_shapes=True, show_layer_names=True)


# ResNet Block
def resnet_block(n_filters, input_layer):

    # Weight initialization
    init = RandomNormal(stddev = 0.02)

    # First convolutional layer
    g = Conv2D(n_filters, (3,3), padding = "same", kernel_initializer = init)(input_layer)
    g = InstanceNormalization(axis = -1)(g)
    g = Activation("relu")(g)

    # Second convolutional layer
    g = Conv2D(n_filters, (3,3), padding = "same", kernel_initializer = init)(g)
    g = InstanceNormalization(axis = -1)(g)

    # Concatenate merge channel-wise with input layer
    g = Concatenate()([g, input_layer])
    return g


# Generator Model
def generator(input_shape = (256, 256, 1), n_resnet = 9):
    # Initialize weights
    init = RandomNormal(stddev=0.02)

    # Source image
    input_image = Input(shape = input_shape)

    # c7s1-64
    g = Conv2D(64, (7,7), padding = "same", kernel_initializer = init)(input_image)
    g = InstanceNormalization(axis = -1)(g)
    g = Activation("relu")(g)

    # d128
    g = Conv2D(128, (3,3), strides=(2,2), padding= "same", kernel_initializer = init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation("relu")(g)

    # d256
    g = Conv2D(256, (3,3), strides=(2,2), padding = "same", kernel_initializer = init)(g)
    g = InstanceNormalization(axis = -1)(g)
    g = Activation("relu")(g)

    # R256
    for i in range(n_resnet):
        g = resnet_block(256, g)

    # u128
    g = Conv2DTranspose(128, (3,3), strides=(2,2), padding = "same", kernel_initializer = init)(g)
    g = InstanceNormalization(axis = -1)(g)
    g = Activation("relu")(g)

    # u64
    g = Conv2DTranspose(64, (3,3), strides=(2,2), padding = "same", kernel_initializer = init)(g)
    g = InstanceNormalization(axis = -1)(g)
    g = Activation("relu")(g)

    # c7s1-3
    g = Conv2D(1, (7,7), padding = "same", kernel_initializer = init)(g)
    g = InstanceNormalization(axis = -1)(g)
    out_image = Activation("tanh")(g)

    # Define model
    model2 = Model(input_image, out_image)
    return model2


# Create model
model2 = generator()

# Summarize model
model2.summary()

# Plot
# plot_model(model2, to_file = "generator_model.png", show_shapes=True, show_layer_names=True)


def composite_model(g_model_1, d_model, g_model_2, input_shape):
    # Set if models are trainable
    g_model_1.trainable = True
    d_model.trainable = False
    g_model_2.trainable = False

    # Discriminator element
    input_gen = Input(shape = input_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)

    # Identity element
    input_id = Input(shape = input_shape)
    output_id = g_model_1(input_id)

    # Forward cycle
    output_f = g_model_2(gen1_out)

    # Backward cycle
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)

    # Define model graph
    model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])

    # Optimization algorithm
    opt = Adam(learning_rate = 0.0002, beta_1 = 0.5)

    # Compile model
    model.compile(loss=["mse", "mae", "mae", "mae"], loss_weights = [1, 5, 10, 10], optimizer = opt)
    return model


# Input shape
input_shape = (256,256,1)
# Generator: A -> B
g_model_AtoB = generator(input_shape)
# Generator: B -> A
g_model_BtoA = generator(input_shape)
# Discriminator: A -> [real/fake]
d_model_A = discriminator(input_shape)
# Discriminator: B -> [real/fake]
d_model_B = discriminator(input_shape)


# Composite: A -> B -> [real/fake, A]
c_model_AtoBtoA = composite_model(g_model_AtoB, d_model_B, g_model_BtoA, input_shape)

# Composite: B -> A -> [real/fake, B]
c_model_BtoAtoB = composite_model(g_model_BtoA, d_model_A, g_model_AtoB, input_shape)


# Select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return X, y

# Generate a batch of images, returns images and targets
def generate_fake_samples(g_model, dataset, patch_shape):
    # generate fake instance
    X = g_model.predict(dataset)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


# Save the generator models to file
def save_models(step, g_model_AtoB, g_model_BtoA):
    # save the first generator model
    filename1 = './models/CycleGAN/g_model_AtoB_%06d.h5' % (step+1)
    g_model_AtoB.save(filename1)
    # save the second generator model
    filename2 = './models/CycleGAN/g_model_BtoA_%06d.h5' % (step+1)
    g_model_BtoA.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


# Generate samples and save as a plot and save the model
def summarize_performance(step, g_model, trainX, name, n_samples=5):
    # select a sample of input images
    X_in, _ = generate_real_samples(trainX, n_samples, 0)
    # generate translated images
    X_out, _ = generate_fake_samples(g_model, X_in, 0)
    # scale all pixels from [-1,1] to [0,1]
    X_in = (X_in + 1) / 2.0
    X_out = (X_out + 1) / 2.0
    X_in = X_in[:,:,:,0]
    X_out = X_out[:,:,:,0]
    # plot real images
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_in[i], cmap='gray')
    # plot translated image
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(X_out[i], cmap='gray')
    # save plot to file
    filename1 = './models/CycleGAN/%s_generated_plot_%06d.png' % (name, (step+1))
    plt.savefig(filename1)
    plt.close()


# Update image pool for fake images
def update_image_pool(pool, images, max_size=30):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif random.random() < 0.5:
            # use image, but don't add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            ix = randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return np.asarray(selected)


# In[ ]:


# train cyclegan models
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA,
dataset):
  # define properties of the training run
  n_epochs, n_batch, = 50, 1

  # determine the output square shape
  n_patch = d_model_A.output_shape[1]

  # unpack dataset
  trainA, trainB = dataset

  # prepare image pool for fakes
  poolA, poolB = list(), list()

  # calculate the number of batches per training epoch
  bat_per_epo = int(len(trainA) / n_batch)

  # calculate the number of training iterations
  n_steps = bat_per_epo * n_epochs

  # manually enumerate epochs
  for i in range(n_steps):
    # select a batch of real samples
    X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
    X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)

    # generate a batch of fake samples
    X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
    X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)

    # update fakes from pool
    X_fakeA = update_image_pool(poolA, X_fakeA)
    X_fakeB = update_image_pool(poolB, X_fakeB)

    # update generator B->A via adversarial and cycle loss
    g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA,
    X_realA, X_realB, X_realA])

    # update discriminator for A -> [real/fake]
    dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
    dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)

    # update generator A->B via adversarial and cycle loss
    g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB,
    X_realB, X_realA, X_realB])

    # update discriminator for B -> [real/fake]
    dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
    dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

    # summarize performance
    print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2,
    dB_loss1,dB_loss2, g_loss1,g_loss2))

    # evaluate the model performance every so often
    if (i+1) % (bat_per_epo * 5) == 0:
      # plot A->B translation
      summarize_performance(i, g_model_AtoB, trainA, 'AtoB')

      # plot B->A translation
      summarize_performance(i, g_model_BtoA, trainB, 'BtoA')

    if (i+1) % (bat_per_epo * 5) == 0:
      # save the models
      save_models(i, g_model_AtoB, g_model_BtoA)

# load image data
dataset = trainA, trainB
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# generator: A -> B
g_model_AtoB = generator(image_shape)
# generator: B -> A
g_model_BtoA = generator(image_shape)
# discriminator: A -> [real/fake]
d_model_A = discriminator(image_shape)
# discriminator: B -> [real/fake]
d_model_B = discriminator(image_shape)
# composite: A -> B -> [real/fake, A]
c_model_AtoB = composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)


# Use of CycleGAN for Image Translation
from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from matplotlib.pyplot import plot, savefig

# select a random sample of images from the dataset
def select_sample(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    return X

# plot the image, the translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
    images = vstack((imagesX, imagesY1, imagesY2))
    titles = ['Real', 'Generated', 'Reconstructed']
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, len(images), 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i,:,:,0], cmap='gray')
        # title
        pyplot.title(titles[i])
    savefig('./figures/CycleGAN/Transformed3to15.png', transparent=True)
    pyplot.show()

# load dataset
A_data, B_data = testA, testB
print('Loaded', A_data.shape, B_data.shape)
# load the models
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('./models/CycleGAN/g_model_AtoB_012250.h5', cust)
model_BtoA = load_model('./models/CycleGAN/g_model_BtoA_012250.h5', cust)
# plot A->B->A
A_real = select_sample(A_data, 1)
B_generated = model_AtoB.predict(A_real)
A_reconstructed = model_BtoA.predict(B_generated)
show_plot(A_real, B_generated, A_reconstructed)
# plot B->A->B
B_real = select_sample(B_data, 1)
A_generated = model_BtoA.predict(B_real)
B_reconstructed = model_AtoB.predict(A_generated)
show_plot(B_real, A_generated, B_reconstructed)

# Evaluation metrics
# A to B
mse_AtoB =  np.mean((A_real - A_reconstructed) ** 2)
psnr_AtoB = 20 * math.log10( 1.0 / math.sqrt(mse_AtoB))
mae_AtoB = np.sum(np.absolute((A_real.astype("float")-A_reconstructed.astype("float"))))
print('The MSE value for A_to_B is: ',mse_AtoB)
print('The PSNR value for A_to_B is: ',psnr_AtoB)
print('The MAE value for A_to_B is: ',mae_AtoB)
# B to A
mse_BtoA =  np.mean((B_real - B_reconstructed) ** 2)
psnr_BtoA = 20 * math.log10( 1.0 / math.sqrt(mse_BtoA))
mae_BtoA = np.sum(np.absolute((B_real.astype("float")-B_reconstructed.astype("float"))))
print('The MSE value for B_to_A is: ',mse_BtoA)
print('The PSNR value for B_to_A is: ',psnr_BtoA)
print('The MAE value for B_to_A is: ',mae_BtoA)



import statistics
import numpy

# 3T to 0.5T, generating 0.5T images
# Load dataset
A_data, B_data = testA, testB
print('Loaded', A_data.shape, B_data.shape)

# Load the models
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('./models/CycleGAN/g_model_AtoB_012250.h5', cust)

# List of values
Avg_PSNR = []
Avg_MSE = []
Avg_MAE = []

for i in range(999):
    # plot A->B->A
    A_real = select_sample(A_data, 1)
    B_generated = model_AtoB.predict(A_real)
    A_reconstructed = model_BtoA.predict(B_generated)

    # Evaluation metrics
    # A to B
    mse_AtoB =  np.mean((A_real - A_reconstructed) ** 2)
    psnr_AtoB = 20 * math.log10( 1.0 / math.sqrt(mse_AtoB))
    mae_AtoB = np.sum(np.absolute((A_real.astype("float")-A_reconstructed.astype("float"))))

    # Append values
    Avg_PSNR.append(psnr_AtoB)
    Avg_MSE.append(mse_AtoB)
    Avg_MAE.append(mae_AtoB)


print('CycleGAN 3T to 1.5T')
print('Average PSNR is: ', statistics.mean(Avg_PSNR))
print('Standard deviation of PSNR is: ', numpy.std(Avg_PSNR))
print('Average MSE is: ', statistics.mean(Avg_MSE))
print('Standard deviation of MSE is: ', numpy.std(Avg_MSE))
print('Average MAE is: ', statistics.mean(Avg_MAE))
print('Standard deviation of MAE is: ', numpy.std(Avg_MAE))


import statistics
import numpy

# 0.5T to 3T, generating 3T images
# Load dataset
A_data, B_data = testA, testB
print('Loaded', A_data.shape, B_data.shape)

# Load the models
cust = {'InstanceNormalization': InstanceNormalization}
model_BtoA = load_model('./models/CycleGAN/g_model_BtoA_012250.h5', cust)

# List of values
Avg_PSNR1 = []
Avg_MSE1 = []
Avg_MAE1 = []

for i in range(999):
    # plot B->A->B
    B_real = select_sample(B_data, 1)
    A_generated = model_BtoA.predict(B_real)
    B_reconstructed = model_AtoB.predict(A_generated)

    # Evaluation metrics
    # B to A
    mse_BtoA =  np.mean((B_real - B_reconstructed) ** 2)
    psnr_BtoA = 20 * math.log10( 1.0 / math.sqrt(mse_BtoA))
    mae_BtoA = np.sum(np.absolute((B_real.astype("float")-B_reconstructed.astype("float"))))

    # Append values
    Avg_PSNR1.append(psnr_BtoA)
    Avg_MSE1.append(mse_BtoA)
    Avg_MAE1.append(mae_BtoA)


print('CycleGAN 1.5T to 3T')
print('Average PSNR is: ', statistics.mean(Avg_PSNR1))
print('Standard deviation of PSNR is: ', numpy.std(Avg_PSNR1))
print('Average MSE is: ', statistics.mean(Avg_MSE1))
print('Standard deviation of MSE is: ', numpy.std(Avg_MSE1))
print('Average MAE is: ', statistics.mean(Avg_MAE1))
print('Standard deviation of MAE is: ', numpy.std(Avg_MAE1))