from .settings import Settings, Section


class ForrestSettings(Settings):
    """
    this class contains settings for random forests
    """
    def __init__(self) -> None:
        self.dataset = Section('data set')
        self.dataset['numCategories'] = 2, 'number of categories of dummy data', int
        self.dataset['dataPath'] = '.', 'the path to data files', str
        self.dataset['dataFiles'] = ['slowpions', 'slowelectrons'], 'the data files used for training and evaluation', list, ['slowpions', 'slowelectrons', 'background']
        self.dataset['features'] = ['cluster'], 'pxd features used for training/evaluation', [str, list], ['event', 'clsNumber', 'cluster', 'clsParameters', 'clsCharge', 'seedCharge', 'clsSize', 'uSize', 'vSize', 'coordinates', 'xPosition', 'yPosition', 'zPosition', 'uvPosition', 'uPosition', 'vPosition', 'pdg']
        self.dataset['trainAmount'] = 50_000, 'the amount of events PER category for training', int
        self.dataset['validAmount'] = 25_000, 'the amount of events PER category for validation', int
        self.dataset['normalize'] = True, 'wether or not to normalize data on import'

        self.forrest = Section('forrest')
        self.forrest['bootstraping'] = False
        self.forrest['booster'] = None, 'boosting algorithm for training', str, [None, 'adaptive', 'gradient']
        self.forrest['retrainFirst'] = False, 'when boosting, the first tree can be droped, created anew and trained again'
        self.forrest['voting'] = 'majority', 'the voting algorithm for the final decision', str, ['majority', 'confidence', 'average', 'median']
        self.forrest['votingWeights'] = None, 'how to weight each trees vote', list

        self.trees = Section('trees', controlParam='numTrees')
        self.trees['numTrees'] = 5
        self.trees['depth'] = [5,5,5,5,5], 'the depth per tree', [int, list]
        self.trees['minSamples'] = [2,2,2,2,2], 'the minimum sample split size'
        self.trees['impurity'] = ['gini', 'gini', 'gini', 'gini', 'gini'], 'impurit measure per tree', [str, list], ['gini', 'entropy', 'mse', 'mae']
        self.trees['leaf'] = ['mode', 'mode', 'mode', 'mode', 'mode'], 'leaf function per tree', [str, list], ['mode', 'mean']
        self.trees['split'] = ['cart', 'cart', 'cart', 'cart', 'cart'], 'split algorithm per tree', [str, list], ['cart', 'id3', 'c45']
        self.trees['percentile'] = [None], "the percentile of feature values to access during training", [int, list]
        self.trees['featSelection'] = [None], "a preselection of features", [None, str], [None, 'variance', 'random', 'mutual', 'kendall', 'anova', 'choice']
        self.trees['featParameter'] = [None], "a threshold feautres need to pass or a number of features to be picked", [None, float, int, list]

        super().__init__()
