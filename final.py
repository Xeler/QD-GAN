from __future__ import print_function, division

from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, Conv2DTranspose, ReLU, Reshape, Conv2D, LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

import matplotlib.pyplot as plt

import time
import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

log_name = "face"

class GAN():
    def __init__(self):
        #Initialize tensorboard logs
        self.tb_g = TensorBoard(log_dir='logs/gen-{}-{:f}'.format(log_name, int(time.time()), batch_size=64, write_grads=True, write_graphs=True, histogram_freq=0))
        self.tb_d = TensorBoard(log_dir='logs/dis-{}-{:f}'.format(log_name, int(time.time()), batch_size=64, write_grads=True, write_graphs=True, histogram_freq=0))

        #images are 28x28 with 1 channel (greyscale)
        self.img_shape = (28, 28, 1)

        #Size of input noise for generator
        self.noise_dim = 100


        #Adam optimizer with lr=0.0002, beta1=0.5
        optimizer = Adam(0.0002, 0.5)


        # Build keras model for discriminator
        self.discriminator = self.build_discriminator()
        # Compile discriminator for use
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build keras model for the generator
        self.generator = self.build_generator()

        # Define input variable (noise)
        z = Input(shape=(self.noise_dim,))
        # Define output variable (generated image)
        img = self.generator(z)

        # Discriminator is set to untrainable. As the model is already compiled, this does not affect
        # the actual model of the discriminator.
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # As discriminator is set to untrainable before compilation, the discriminator will
        # remain untrainable, and only the the generator is adjusted during training of the combined model.
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        #Attach models to tensorboard logs
        self.tb_g.set_model(self.combined)
        self.tb_d.set_model(self.discriminator)


    def build_generator(self):
        
        #The model is a sequential keras model.
        model = Sequential()

        #Input noise is projected to a dense layer of desired size
        model.add(Dense(7*7*64, input_shape=(self.noise_dim,)))
        model.add(BatchNormalization())
        model.add(ReLU())
        #Reshape to 64 samples of 7x7 noise
        model.add(Reshape((7,7,64)))

        #Layer 1
        model.add(Conv2DTranspose(64, (5,5), strides=1, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(ReLU())

        #Layer 2 (Dimensions increases by factor 2)
        model.add(Conv2DTranspose(32, (5,5), strides=2, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(ReLU())

        #Layer 3 (output) (Dimensions increases by factor 2, resulting output is 1 sample of 28x28)
        model.add(Conv2DTranspose(1, (5,5), strides=2, padding="same", use_bias=False))
        model.add(Activation("tanh"))

        #Print summary of the model
        model.summary()
        
        #Define input and outputs, and return the resulting model
        noise = Input(shape=(self.noise_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()
        #Layer 1, downsampling
        model.add(Conv2D(64, (3,3), input_shape=(28,28,1), strides=2, padding="same"))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        #Layer 2, downsampling
        model.add(Conv2D(128, (3,3), strides=2, padding="same"))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        
        #Flatten & classify (output)
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        #Inputs & outputs
        img = Input(shape=self.img_shape)
        validity = model(img)

        #Model is returned
        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train = np.load("../full_numpy_bitmap_face.npy")
        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        #Reshape to actual image dimensions (image array is currently 1 dimensional)
        X_train = X_train.reshape((X_train.shape[0], int(X_train.shape[1]/28), 28))
        
        #Add 3rd dimension (1 channel)
        X_train = np.expand_dims(X_train, axis=3)
        
        # Training values fit to batch_size (valid images are presented by 1, generated images are represted by 0)
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # Get random indexes from the dataset
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            # Select the images used for training
            imgs = X_train[idx]

            # Generate input noise for the generator
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            # Generate batch of images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator on valid & generated images. Loss & Accuracy is returned & saved for logging
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # Train the generator to fool the discriminator, using the stacked model.
            # As the discriminator is untrainable, only the generator is adjusted.
            g_loss = self.combined.train_on_batch(noise, valid)

            # If at sample interval, save images
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                log = {}
                log["loss_real"] = d_loss_real[0]
                log["loss_fake"] = d_loss_fake[0]
                log["loss"] = d_loss[0]
                log["acc"] = d_loss[1] * 100
                self.tb_d.on_epoch_end(epoch, log)
                log = {}
                log["loss"] = g_loss
                self.tb_g.on_epoch_end(epoch, log)

    def sample_images(self, epoch):
        # Plot grayscale images. Code found on github (link missing)
        r, c = 5, 5 #rows / columns

        noise = np.random.uniform(0, 1, (r * c, self.noise_dim))
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
        fig.savefig("img_face/%d.png" % epoch)
        plt.close()


#If the file was executed and not included:
if __name__ == '__main__':
    #Define gan object
    gan = GAN()
    #Train gan using defined settings
    gan.train(epochs=200000, batch_size=64, sample_interval=200)
    #When training is finished, save the generator for future use.
    gan.generator.save("gen-{}".format("face"))