from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tflearn
from tflearn.data_utils import shuffle

# Data loading and preprocessing
from tflearn.datasets import cifar10
(X, Y), (X_test, Y_test) = cifar10.load_data(one_hot = True, flat = True)
X, Y = shuffle(X, Y)


# Building the encoder
encoder = tflearn.input_data(shape=[None, 32*32*3])
encoder = tflearn.fully_connected(encoder, 512)
encoder = tflearn.fully_connected(encoder, 512)
encoder = tflearn.fully_connected(encoder, 64, activation='tanh')

# Binarize layer
binary = encoder

# Building the decoder
decoder = tflearn.fully_connected(binary, 512)
decoder = tflearn.fully_connected(decoder, 512)
decoder = tflearn.fully_connected(decoder, 32*32*3)

# Regression, with mean square error
net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.001,
                         loss='mean_square', metric=None)

# Training the auto encoder
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, X, n_epoch=1, validation_set=(X_test, X_test),
          run_id="compression_fc_encoder_cifar10", batch_size=128)



"""
print("\nTest encoding of X[0]:")
# New model, re-using the same session, for weights sharing
encoding_model = tflearn.DNN(encoder, session=model.session)
print(encoding_model.predict([X[0]]))

# Testing the image reconstruction on new data (test set)
print("\nVisualizing results after being encoded and decoded:")

# Applying encode and decode over test set
encode_decode = model.predict(X_test[0:10, :])
# Compare original images with their reconstructions
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    a[0][i].imshow(np.reshape(X_test[i], (32, 32,3)))
    a[1][i].imshow(np.reshape(encode_decode[i], (32, 32,3)))
f.show()
plt.draw()
plt.waitforbuttonpress()
"""
