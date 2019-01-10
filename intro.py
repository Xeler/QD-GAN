#kunne det evt være interessant at udføre forsøget udelukkende på ikke-genkendte tegninger, og se om den alligevel vil generere succesfulde tegninger+



#http://colah.github.io/posts/2015-08-Understanding-LSTMs/



#https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
#https://carpedm20.github.io/faces/ illustrerer latent space sindsygt godt <3

#https://www.youtube.com/watch?v=ywinX5wgdEU lstm   


#We can achieve this by wrapping the entire CNN input model (one layer or more) in a TimeDistributed layer. This layer achieves the desired outcome of applying the same layer or layers multiple times. In this case, applying it multiple times to multiple input time steps and in turn providing a sequence of “image interpretations” or “image features” to the LSTM model to work on.

import ndjson
import math
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.layers import Flatten, Conv2D, LSTM, Dense, Input, InputLayer, ConvLSTM2D, ReLU, MaxPooling3D, MaxPooling2D, MaxPool1D, Reshape, TimeDistributed, Conv1D
from keras.optimizers import Adam
from keras.models import Model, Sequential


#from keras.utils import 



from itertools import compress
from random import shuffle



latent_dim = (4, 100)

drawing_dim = (4, 30, 2, 1)




def draw(drawing):
    for seq in drawing:
        plt.plot(seq[0], seq[1])

    return True

def train(epochs, batch_size=100, sample_interval=50):

    # Load the dataset
    X_train = fetch_data(batch_size)
    # Rescale 0 to 1
    #X_train = X_train / 255.
    
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

        noise = np.random.normal(0, 1, (batch_size, 4, 100))

        # Generate a batch of new images
        gen_imgs = generator.predict(noise)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator (to have the discriminator label samples as valid)
        g_loss = combined.train_on_batch(noise, valid)

        # Plot the progress
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        # If at save interval => save generated image samples
        if epoch % sample_interval == 0:
            print(gen_imgs[0])




def build_descriminator():
    model = keras.Sequential()
    model.add(InputLayer(batch_input_shape=(100, 4, 30, 2, 1)))
    print(model.output_shape)

    #"causal" results in causal (dilated) convolutions, e.g. output[t] does not depend on input[t + 1:]. A zero padding is used such that the output has the same length as the original input. Useful when modeling temporal data where the model should not violate the temporal order. See WaveNet: A Generative Model for Raw Audio, section 2.1.
    model.add(ConvLSTM2D(32, (2,2), return_sequences=True, padding="same", dropout=0.1))
    model.add(MaxPooling3D((2,2,1)))
    model.add(ReLU())
    print(model.output_shape)
    
    
    model.add(ConvLSTM2D(24, (2,1)))
    print(model.output_shape)

    model.add(MaxPooling2D())
    model.add(ReLU())
    print(model.output_shape)

    model.add(Conv2D(16, (2,1)))
    model.add(ReLU())
    print(model.output_shape)
    model.add(Flatten())
    print(model.output_shape)
    model.add(Dense(1, activation="sigmoid"))
    print(model.output_shape)
    
    #


    #Hvis input_shape er None, kan timesteps være variabel
#    model.add(TimeDistributed(Conv2D(32, (2, 2), data_format="channels_last")))
#    model.add(TimeDistributed(Conv2D(16, (2, 2))))
    

    img = Input(shape=(4, 30, 2, 1))
    validity = model(img)

    return Model(img, validity)



def get_mask():
    length = np.zeros(len(data))


    '''
    Beregn maks antal linjer af alle strøg hver tegning. 
    '''

    for i, r in enumerate(data):
        length[i] = np.max([len(_[0]) for _ in r["drawing"]])

    #plt.plot(length)
    #np.size(length[length>30])
    #er = 19270, dvs under 10 procent af dataen anvender mere end 30 linjer per strøg

    #Vi oprtetter en maske til vores data som er over dette antal:

    mask = length<=30

    '''
    Beregn antal strøg / "tryk"
    '''
    for i, r in enumerate(data):
        length[i] = np.size(r["drawing"],0)


    #np.size(length[length>5]) = 19.7k, altså under 10% af tegninger bruger mere end 5 strøg
    #plt.plot(length)
    #plt.show()
    #mask er nu kombineret af begge.


    mask = [_<=4 and mask[i] for i, _ in enumerate(length)]

    return mask







idx = 0


def fetch_data(batch_size=1):
    global idx
    if idx + batch_size > len(f_data):
        idx = 0
    
    
    items = items = np.zeros((batch_size, 4, 30, 2, 1))
    for k,v in enumerate(f_data[idx:idx+batch_size]):
        for i, s in enumerate(v["drawing"]):
            for j, m in enumerate(s[0]):
                items[k][i][j][0] = m
            
            for j, m in enumerate(s[1]):
                items[k][i][j][1] = m
            
        
    

            



    idx = idx + batch_size
    return items



def build_generator():

    model = Sequential()
    ##Gen model mangler

    model.add(LSTM(128, input_shape=latent_dim, return_sequences=True))
    print(model.output_shape)
    model.add(ReLU())

    model.add(LSTM(256, return_sequences=True))
    model.add(ReLU())
    print(model.output_shape)

    model.add(TimeDistributed(Reshape((64,4,1))))
    print(model.output_shape)
    model.add(TimeDistributed(Conv2D(60, (2,2), padding="same")))
    model.add(ReLU())

    print(model.output_shape)
    model.add(TimeDistributed(MaxPooling2D((2,2))))
    print(model.output_shape)

    model.add(TimeDistributed(Flatten()))
    print(model.output_shape)
    model.add(TimeDistributed(Dense(30*2)))
    model.add(TimeDistributed(Reshape((30,2,1))))    

    model.summary()

    noise = Input(shape=latent_dim)
    img = model(noise)

    return Model(noise, img)









with open('simplified_banana.ndjson') as f:
    data = ndjson.load(f)

mask = get_mask()


f_data = list(compress(data, mask))




        # Build the generator
generator = build_generator()

#np.max([len(_[0]) for _ in data[0]["drawing"]])
# indlæs data fra ndjson fil





optimizer = Adam(0.0002, 0.5)


# Build and compile the discriminator
discriminator = build_descriminator()
discriminator.compile(loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])

discriminator.summary()

discriminator.trainable = False


# The generator takes noise as input and generates imgs
z = Input(shape=latent_dim)
img = generator(z)

# The discriminator takes generated images as input and determines validity
validity = discriminator(img)

# The combined model  (stacked generator and discriminator)
# Trains the generator to fool the discriminator
combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)



print("ok!!!")




train(10002)






print("a")






#https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
#
#"Generally, we do not need to access the cell state unless we are developing sophisticated models where subsequent layers
#may need to have their cell state initialized with the final cell state of another layer, such as in an encoder-decoder model."
#
#
#
#









'''
model = keras.Sequential()
#Hvis input_shape er None, kan timesteps være variabel
model.add(keras.layers.LSTM(32, return_sequences=True, input_shape=(None, 5)))
model.add(keras.layers.LSTM(8, return_sequences=True))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(2, activation='sigmoid')))

print(model.summary(90))


model.compile(loss='categorical_crossentropy',
              optimizer='adam')

model.fit_generator()
'''