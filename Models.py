import optparse
import sys
from MultiClassNeuralNetowrkModel import mnist_neural_network_model
from FirstModel import FirstModel

MODEL_CHOICES=("FirstModel", "MNIST Neural Network")
DEFAULT_CHOICE="FirstModel"
MODEL_MODES=("Train", "Test", "Train&Test")
MODEL_FUNCTIONS={"FirstModel": FirstModel, "MNIST Neural Network": mnist_neural_network_model}


def readCommand(argv):
    parser = optparse.OptionParser(description = 'Run, Train, Present, and Store Models')
    parser.set_defaults(modelMode="Train", loadModel=False, saveModel=False, presentPerformance=True, muteOutput=False)
    parser.add_option('-m', '--model',
                      type='choice', choices=MODEL_CHOICES,
                      dest='modelName',
                      help = 'Which model to run')
    parser.add_option('-r', '--learning-rate',
                        type='float',
                        dest='learningRate',
                        help ='The learning rate to give the model')
    parser.add_option('-b', '--batchSize',
                        type='int',
                        dest='batchSize',
                        help='The batch size for this model')
    parser.add_option('-e', '--epochs',
                        type='int',
                        dest='epochs',
                        help='The amount of epochs to train the model')
    parser.add_option('-o', '--mode',
                        type='choice', choices=MODEL_MODES,
                        dest='modelMode',
                        )
    parser.add_option('-v', '--validationSplit',
                        type='float',
                        dest='validationSplit',
                        help='The split of the test set for validation')
    parser.add_option('-s', '--save-model',
                        action='store_true',
                        dest='saveModel')
    parser.add_option('-l', '--load-model',
                        action='store_true',
                        dest='loadModel')

    (options, args) = parser.parse_args(argv)
    if not options.modelName:
        parser.error("Model name not given")
    return options
if __name__ == "__main__":
    options = readCommand(sys.argv[1:])
    print(options)
    MODEL_FUNCTIONS[options.modelName](options)