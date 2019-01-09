#kunne det evt være interessant at udføre forsøget udelukkende på ikke-genkendte tegninger, og se om den alligevel vil generere succesfulde tegninger+



#http://colah.github.io/posts/2015-08-Understanding-LSTMs/



#https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
#https://carpedm20.github.io/faces/ illustrerer latent space sindsygt godt <3


#We can achieve this by wrapping the entire CNN input model (one layer or more) in a TimeDistributed layer. This layer achieves the desired outcome of applying the same layer or layers multiple times. In this case, applying it multiple times to multiple input time steps and in turn providing a sequence of “image interpretations” or “image features” to the LSTM model to work on.

import ndjson
import math
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.layers import Conv2D, TimeDistributed, LSTM, Dense, Input
from keras.optimizers import Adam
from keras.models import Model, Sequential


#from keras.utils import 



from itertools import compress
from random import shuffle




def draw(drawing):
    for seq in drawing:
        plt.plot(seq[0], seq[1])

    return True



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


    mask = [_<=5 and mask[i] for i, _ in enumerate(length)]

    return mask

idx = 0

#np.max([len(_[0]) for _ in data[0]["drawing"]])
# indlæs data fra ndjson fil
with open('simplified_banana.ndjson') as f:
    data = ndjson.load(f)

mask = get_mask()


f_data = list(compress(data, mask))

shuffle(f_data)



optimizer = Adam(0.0002, 0.5)

# Build and compile the discriminator
discriminator = build_descriminator()
discriminator.compile(loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])






def build_generator(self):

    model = Sequential()
    ##Gen model mangler

    model.add(Dense(256, input_dim=self.latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(self.img_shape), activation='tanh'))
    model.add(Reshape(self.img_shape))

    model.summary()

    noise = Input(shape=(self.latent_dim,))
    img = model(noise)

    return Model(noise, img)



def fetch_data(batch_size=1):
    global idx
    if idx + batch_size > len(f_data):
        idx = 0
    
    
    items = items = np.zeros((batch_size, 5, 30, 2))
    for k,v in enumerate(f_data[idx:idx+batch_size]):
        for i, s in enumerate(v["drawing"]):
            for j, m in enumerate(s[0]):
                items[k][i][j][0] = m
            
            for j, m in enumerate(s[1]):
                items[k][i][j][1] = m
            
        
    

            



    idx = idx + batch_size
    return items


#https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
#
#"Generally, we do not need to access the cell state unless we are developing sophisticated models where subsequent layers
#may need to have their cell state initialized with the final cell state of another layer, such as in an encoder-decoder model."
#
#
#
#






def build_descriminator():
    model = keras.Sequential()
    #Hvis input_shape er None, kan timesteps være variabel
    model.add(TimeDistributed(Conv2D(32, (2, 2)), input_shape=(5, 30, 2)))
    model.add(TimeDistributed(Conv2D(16, (2, 2))))
    
    model.add(LSTM(16, return_sequences=True,))
    model.add(LSTM(8, return_sequences=True))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=(5,  30, 2))
    validity = model(img)

    return Model(img, validity)




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