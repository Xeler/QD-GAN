from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Conv2DTranspose, ReLU, Reshape, Conv2D, LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt

import sys
import numpy as np

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.img_shape = (self.img_rows,  self.img_cols, 1)
        self.latent_dim = 100

        optimizer = Adam(0.0002)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

    
        model.add(Dense(7*7*64, input_dim=self.latent_dim, activation="relu"))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Reshape((7,7,64)))

        model.add(Conv2DTranspose(64, (5,5), padding="same", use_bias=False))
        model.add(BatchNormalization())
        print(model.output_shape) 
        model.add(ReLU())
        print(model.output_shape)
        model.add(Conv2DTranspose(32, (5,5), strides=(2,2), padding="same", use_bias=False))
        model.add(BatchNormalization())

        model.add(Conv2DTranspose(1, (5,5), strides=(2,2), padding="same", use_bias=False))

        print(model.output_shape)

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()
        model.add(Conv2D(64, (5,5), input_shape=(28,28,1), strides=(2,2), padding="same"))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))
        model.add(Conv2D(128, (5,5), strides=(2,2), padding="same"))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))
        model.add(Flatten())
        print(model.output_shape)
        model.add(Dense(1, activation='sigmoid'))
        print(model.output_shape)
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train = np.load("full_numpy_bitmap_face.npy")
        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = X_train.reshape((X_train.shape[0], int(X_train.shape[1]/28), 28))
        
        X_train = np.expand_dims(X_train, axis=3)
        
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #  Train Generator
            # ---------------------


            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5 #rows / columns

        noise = np.random.uniform(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        
        # Rescale images 0 - 1 (-1;1 => 0;1)
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
gan.train(epochs=300000, batch_size=64, sample_interval=200)