from .settings import Settings, Section


class NetworkSettings(Settings):
    """
    this class contains parameters for setting up neural networks
    """
    def __init__(self) -> None:
        self.dataset = Section('data set')
        self.dataset['numCategories'] = 2, 'number of categories of dummy data', int
        self.dataset['dataPath'] = '.', 'the path to data files', str
        self.dataset['dataFiles'] = ['slowpions', 'slowelectrons'], 'the data files used for training and evaluation', list, ['slowpions', 'slowelectrons', 'background']
        self.dataset['features'] = ['cluster'], 'pxd features used for training/evaluation', [str, list], ['event', 'clsNumber', 'cluster', 'clsParameters', 'clsCharge', 'seedCharge', 'clsSize', 'uSize', 'vSize', 'coordinates', 'xPosition', 'yPosition', 'zPosition', 'uvPosition', 'uPosition', 'vPosition', 'pdg']
        self.dataset['trainAmount'] = 50_000, 'the amount of events PER category for training', int
        self.dataset['validAmount'] = 25_000, 'the amount of events PER category for validation', int
        self.dataset['batchSize'] = 64, 'the chunk size at which will be trained, for validation this is twice as much', int
        self.dataset['kFold'] = 4, 'the split fraction for cross validation during training', int
        self.dataset['normalize'] = False, 'wether or not to normalize data on import'

        self.convolution = Section('convolution', controlParam='convolutions', completionMode=1, identify=True)
        self.convolution['convolutions'] = {}, 'a dict with keys identifing the parallel running convolutions, behind each key is an int setting the number of consecutive convolutions', [None], [None], 0
        self.convolution['activationConv'] = {}, 'a dict with keys, identifing the parallel running convolutions, behind each key is an str setting the activation function', [None], [None], ['tanh']
        self.convolution['channels'] = {}, 'a dict with keys identifing the parallel running convolutions, behind each key is a list setting the number of channels per consecutive convolution', [None], [None], [(1,3)]
        self.convolution['kernelSize'] = {}, 'a dict with keys identifing the parallel running convolutions, behind each key is a tuple/int setting the filter/kernel size per consecutive convolution', [None], [None], [(3,3)]
        self.convolution['convStride'] = {}, 'a dict with keys identifing the parallel running convolutions, behind each key is a tuple/int setting the filter/kernel step size per consecutive convolution', [None], [None], [(1,1)]
        self.convolution['padding'] = {}, 'a dict with keys identifing the parallel running convolutions, behind each key is a list setting the amout of padding per consecutive convolution', [None], [None], [(0,0)]
        self.convolution['pooling'] = {}, 'a dict with keys identifing the parallel running convolutions, behind each key is a list setting if there is max-pooling after each consecutive convolution', [None], [None], [False]
        self.convolution['poolingSize'] = {}, 'a dict with keys identifing the parallel running convolutions, behind each key is a list setting how many pixels should be max-pooled after each consecutive convolution', [None], [None], [(2,2)]
        self.convolution['poolStride'] = {}, 'a dict with keys identifing the parallel running convolutions, behind each key is a tuple/int setting the filter/kernel step size per consecutive convolution', [None], [None], [(2,2)]
        self.convolution['convNorm'] = {}, 'a dict with keys identifing the parallel running convolutions, behind each key is a bool setting if the convolution output should be normalized', [None], [None], [False]

        self.linear = Section('linear', controlParam='numLayers', completionMode=1)
        self.linear['numLayers'] = 3, 'number of layers', int
        self.linear['activationLin'] = ['tanh', 'tanh', 'tanh', 'tanh'], 'the activation function for ALL neurons, but the last', [str, list], ['lrelu', 'elu', 'relu', 'sigmoid', 'tanh', 'id', 'softplus', 'softmax', 'softsign']
        self.linear['numNeurons'] = [81, 81, 81, 81], 'number of neurons on each layer', [int, list]
        self.linear['dropoutRate'] = [0.35, 0.35, 0.35, 0.35], 'dropout rate per layer', [float, list]
        self.linear['linearNorm'] = [False, False, False, False], 'setting a batchnorm per layer', [bool, list]

        self.training = Section('training')
        self.training['epochs'] = 50, 'the maximum amount of epochs, after which the training will be ended'
        self.training['optim'] = 'adam', 'the optimiser used for training the network', str, ['rmsprop', 'adam', 'adagrad', 'adadelta', 'sgd', 'momentum', 'nesterov']
        self.training['lossFunction'] = 'entropy', 'picking the loss function, it is important to pick the right function for the problem', str, ['entropy', 'nllloss', 'mseloss', 'maeloss', 'focalloss']
        self.training['learningRate'] = 0.001, 'the learning rate at which the network will learn'
        self.training['momentum'] = 0.9, 'momentum of the optimizer'
        self.training['scheduler'] = None, 'the type of learn rate scheduler', [str, None], [None, 'stepped', 'exponential', 'else']
        self.training['decayrate'] = 0.9, 'the factor by which the learnin rate will be multiplied after every step'
        self.training['stepSize'] = 10, 'the amount of steps to be taken to adjust the learning rate'

        super().__init__()