# TensorflowFun
My Exploration With Tensorflow!

Through out this repo there will be many different models being created. 

`Models.py` was created to help with creation, testing, tuning, loading, and saving models.

## Models.py
This script takes in command line inputs such as the model to use and the hyperparamters to be passed. With these inputs, the given model is executed.

### Command Line Inputs
Here are the types of inputs to pass to this script:
- `--modelName` or `-m` (**string**) (REQURED)
: Represents which model to excute (Can only choose models in the list of choices)
- `--mode` or `-o` (**string**)
: Represents how to interact with the given mode (Current Choices are: `Train`, `Test`, `Train&Test`)
- `--learningRate` or `-l` (**float**)
: Represents the learning rate to be passed to the model
- `--batchSize` or `-b` (**int**)
: Represents the batch size to be passed to the model (the amount of training examples to work through before updating weights)
- `--epochs` or `-e` (**int**)
: Represents the epochs to be passed to the model (the amount of full run throughs of the training data)
- `--validationSplit` or `-v` (**float**)
: Represents the precentage of the test data to be used as a validation set 
- `--load-model` (**boolean flag**)
: Tells the model script to load a saved version of the model
- `--save-model` (**boolean flag**)
: Tells the model script to save the model at the end of execution

### Adding Models to Models.py
To add models to `Models.py` there are a couple of things to change to `Models.py`

#### Steps:
1. Add your model name to the `MODEL_CHOICES` list
2. Add your model function as a key-value entry in `MODEL_FUNCTIONS`. With the model name you added to `MODEL_CHOICES` as the key and the model's function as the value (Take a look at how the MNIST Neural Network Model is added)
3. In your model function, make sure to take a single arguement (call it argv if you'd like). This will hold the values of inputs passed into the command line. To access you can do argv.{input name} (Ex: argv.learningRate). The inputs will follow camelCase style.
