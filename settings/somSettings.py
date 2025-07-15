from .settings import Settings, Section


class SOMSettings(Settings):
    """
    hold the settings for self organizing maps
    """
    def __init__(self) -> None:
        self.dataset = Section('data set')
        self.dataset['numCategories'] = 2, 'number of categories of dummy data', int
        self.dataset['dataPath'] = '.', 'the path to data files', str
        self.dataset['dataFiles'] = ['slowpions', 'slowelectrons'], 'the data files used for training and evaluation', list, ['slowpions', 'slowelectrons', 'background']
        self.dataset['features'] = ['clsParameters'], 'pxd features used for training/evaluation', [str, list], ['event', 'clsNumber', 'cluster', 'clsParameters', 'clsCharge', 'seedCharge', 'clsSize', 'uSize', 'vSize', 'coordinates', 'xPosition', 'yPosition', 'zPosition', 'uvPosition', 'uPosition', 'vPosition', 'pdg']
        self.dataset['trainAmount'] = 5_000, 'the amount of events PER category for training', int
        self.dataset['validAmount'] = 2_500, 'the amount of events PER category for validation', int
        self.dataset['batchSize'] = 64, 'the chunk size at which will be trained, for validation this is twice as much', int
        self.dataset['normalize'] = True, 'wether or not to normalize data on import'

        self.som = Section('som')
        self.som['gridSize'] = [15,15], 'the size of the grid', list
        self.som['topology'] = 'rectangular', 'sets the topology of the som', str, ['rectangular', 'hexagonal']
        self.som['neighborhood'] = 'gaussian', 'the neighborhood function', str, ['gaussian', 'mexicanhat', 'bubble', 'linear', 'cosine', 'cauchy', 'epanechnikov']
        self.som['scale'] = 1, 'the steps size or radius of the neighborhood function', int

        self.training = Section('training')
        self.training['gridSteps'] = 2, 'the steps size or radius of the neighborhood function', int
        self.training['decreaseEvery'] = 25, 'every ... epoch the gridStep size will be decreased by 1', int
        self.training['epochs'] = 100, 'training iterations', int
        self.training['learningRate'] = 0.01, 'scaling factor for adjunsting weights', float
        self.training['scheduler'] = None, 'the type of learn rate scheduler', [str, None], [None, 'stepped', 'exponential', 'else']
        self.training['decayrate'] = 0.9, 'the factor by which the learnin rate will be multiplied after every step'
        self.training['stepSize'] = 10, 'the amount of steps to be taken to adjust the learning rate'

        super().__init__()