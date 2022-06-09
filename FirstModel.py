import tensorflow as tf;
import numpy as np;

def FirstModel(options):

    # Setup random dataset
    x = np.arange(1, 101, step=0.1)
    y = [X**2 for X in x]

    x = tf.cast(tf.constant(x), dtype=tf.float32)
    y = tf.cast(tf.constant(y), dtype=tf.float32)

    # Created a simple model with an single feature and single output
    model = tf.keras.models.Sequential()

    # Output Layer
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    learning_rate = 0.03
    ecpoch = 100
    batch_size = None
    

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
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate),
                loss = "mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.fit(x,y, epochs= ecpoch)

if __name__ == "__main__":
    FirstModel(None)