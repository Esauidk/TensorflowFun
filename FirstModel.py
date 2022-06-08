import tensorflow as tf;
import numpy as np;

def main():

    # Setup random dataset
    x = np.arange(1, 101, step=0.1)
    y = [X**2 for X in x]

    x = tf.cast(tf.constant(x), dtype=tf.float32)
    y = tf.cast(tf.constant(y), dtype=tf.float32)

    # Created a simple model with an single feature and single output
    model = tf.keras.models.Sequential()

    # Output Layer
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    learningRate = 0.03
    ecpoch = 100
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learningRate),
                loss = "mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.fit(x,y, epochs= ecpoch)

if __name__ == "__main__":
    main()