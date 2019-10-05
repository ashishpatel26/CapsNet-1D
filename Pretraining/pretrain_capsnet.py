# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 09:31:07 2018

@author: Karush
"""

"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""
#import json
#
#with open("keras.json") as json_file:
#		s2 = json.load(json_file)
#with open("keras.json",'w') as json_file:
#    json.dump(s1,json_file) 


import numpy as np
import json
from keras import layers, models, optimizers
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

#%%

def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv1D(filters=256, kernel_size=12, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=5, n_channels=20, kernel_size=12, strides=2, padding='valid')
    
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=10, num_routing=routings,
                             name='digitcaps')(primarycaps)
    
    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps) 

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    dec_3 = decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid', input_dim=10*n_class))
#    dec_1 = decoder.add(layers.Dense(16, activation='relu', input_dim=10*n_class))
#    dec_2 = decoder.add(layers.Dense(16, activation='relu'))
#    decoder.add(layers.Dropout(0.40))
#    dec_3 = decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    dec_4 = decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked)]) #masked_by_y
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 10))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, X_train, y_train, X_test, y_test, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr, amsgrad=True),
                  loss=[margin_loss, 'categorical_crossentropy'],
                  loss_weights=[1, args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

#    path = r'C:\Users\Karush\.spyder-py3\CapsNet_Weights5'
#    for j in range(1,11):
#        model.load_weights(path+r"\weights-improvement-"+str(j)+".hdf5")
#        filters = model.layers[8].get_weights()[0]
#        np.shape(filters)
#        filters = np.reshape(filters, (int(200*4),int(9324/4)))
#        plt.figure()
#        plt.imshow(filters)
#        plt.savefig(r'C:\Users\Karush\.spyder-py3\CapsNet_Activations5\c'+str(j)+'.png')

    model.load_weights('new_capsnet20.h5')
    # Training without data augmentation:
    hist = model.fit([X_train, y_train], [y_train, X_train], batch_size=args.batch_size, epochs=args.epochs,
               shuffle=True)
    
    model.save_weights('new_capsnet30.h5')
    ## Saving model to JSON
#    model_json = model.to_json()
#    with open("5capsnet_50.json", "w") as json_file:
#        json_file.write(model_json)
#    
#    ## Saving model data to JSON
#    with open('5capsnet_50.json', 'w') as f:
#        json.dump(hist.history, f)
#    
#    model.save_weights('5capsnet_50.h5')

    return model


def test(model, X_test, y_test, args):
    y_pred, x_recon = model.predict(X_test, batch_size=20)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])
    return y_pred

def manipulate_latent(model, X_train, y_train, X_test, y_test, args):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = X_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)


def load_data():
    # the data, shuffled and split between train and test sets
    new_data = np.load('11_data.npy')
    X = np.load('final_data.npy')
    X = X[:,0:len(new_data[1,:,1]),:]
    
    X_train = np.zeros((0,1032,9))
    y_train = np.zeros((0,1))
    act = [3,5,6,11,12,14,15,16,17,19]
    
    for j in range(0,len(act)):
        X_train = np.concatenate((X_train,new_data[(act[j]-1)*10:(act[j])*10,:,:]),axis=0)
        X_train = np.concatenate((X_train,X[(act[j]-1)*100:(act[j])*100,:,:]),axis=0)
        y_train = np.concatenate((y_train,j*np.ones((len(new_data[(act[j]-1)*10:(act[j])*10,:,:]),1))),axis=0)
        y_train = np.concatenate((y_train,j*np.ones((len(X[(act[j]-1)*100:(act[j])*100,:,:]),1))),axis=0)
    
    y_train = to_categorical(y_train)
    X_test = X_train
    y_test = y_train
    return X_train, y_train, X_test, y_test

#%%
if __name__ == "__main__":
    import argparse

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on Gestures.")
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--lr', default=0.0001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    # load data
    X_train,y_train,X_test,y_test = load_data()
    
    
    # define model
    model, eval_model, manipulate_model = CapsNet(input_shape=X_train[1,:,:].shape,
                                                  n_class=10,
                                                  routings=args.routings)
    model.summary()

    # train or test
    train(model, X_train, y_train, X_test, y_test, args)
    pred = test(eval_model, X_test, y_test, args)




