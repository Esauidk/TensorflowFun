from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import sys

# The following lines adjust the granularity of reporting. 
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# The following line improves formatting when ouputting NumPy arrays.
np.set_printoptions(linewidth = 200)

def mnist_neural_network_model(options):
    (x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

    #plt.imshow(x_train[2917])
    #plt.show()

    x_train_normalized = x_train / 255
    x_test_normalized = x_test / 255
   
    learning_rate = 0.003
    epochs = 50
    batch_size = 4000
    validation_split = 0.2
    
    saveModel = False
    loadModel = False
    if options is not None:
        print(options)
        if options.learningRate is not None:
            learning_rate = options.learningRate
        if options.batchSize is not None:
            batch_size = options.batchSize
        if options.epochs is not None:
            saveModel = options.epochs
        loadModel = options.loadModel
        saveModel = options.saveModel

    model = None
    # Create the Model
    if not loadModel:
        model = create_model(learning_rate)
    else:
        model = tf.keras.models.load_model('SavedModels/MNISTNeuralNetworkModel')

    # Train the model on the normalized training set.
    epochs, hist = train_model(model, x_train_normalized, y_train, 
                            epochs, batch_size, validation_split)

    # Plot a graph of the metric vs. epochs.
    list_of_metrics_to_plot = ['accuracy']
    plot_curve(epochs, hist, list_of_metrics_to_plot)

    # Evaluate against the test set.
    print("\n Evaluate the new model against the test set:")
    model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)

    if saveModel:
        model.save('SavedModels/MNISTNeuralNetworkModel')


def plot_curve(epochs, hist, list_of_metrics):
  """Plot a curve of one or more classification metrics vs. epoch."""  
  # list_of_metrics should be one of the names shown in:
  # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics  

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")

  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)

  plt.legend()
  plt.show()

def create_model(learning_rate):
    # Creating a Sequential Model
    model = tf.keras.models.Sequential()

    # Input Layer (Flattens the 2D Image to a 1D array of (x,y) values)
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

    # First Hidden Layer: Hoping 255 can understand the relationship between the orignal 0-255 values
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))

    # Dropout layer: Since this data set consist of only 0-9, we don't want to overfit to this data set 
    model.add(tf.keras.layers.Dropout(rate=0.2))

    # Output Layer: We have 10 different possible results, need to use softmax for this
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))   

    # Compile the model with the given learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
    return model

def train_model(model, train_features, train_label, epochs,
                batch_size=None, validation_split=0.1):
    snap = model.fit(x=train_features, y=train_label, batch_size= batch_size, 
                    shuffle=True, epochs=epochs, validation_split=validation_split)
    epochs = snap.epoch
    snip = pd.DataFrame(snap.history)

    return (epochs,snip)



if __name__ == "__main__":
    mnist_neural_network_model(None)