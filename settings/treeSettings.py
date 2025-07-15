from .settings import Settings, Section


class TreeSettings(Settings):
    """
    settings for decision tree
    """
    def __init__(self) -> None:
        self.dataset = Section('data set')
        self.dataset['numCategories'] = 2, 'number of categories of dummy data', int
        self.dataset['dataPath'] = '.', 'the path to data files', str
        self.dataset['dataFiles'] = ['slowpions', 'slowelectrons'], 'the data files used for training and evaluation', list, ['slowpions', 'slowelectrons', 'background']
        self.dataset['features'] = ['clsParameters'], 'pxd features used for training/evaluation', [str, list], ['event', 'clsNumber', 'cluster', 'clsParameters', 'clsCharge', 'seedCharge', 'clsSize', 'uSize', 'vSize', 'coordinates', 'xPosition', 'yPosition', 'zPosition', 'uvPosition', 'uPosition', 'vPosition', 'pdg']
        self.dataset['trainAmount'] = 5_000, 'the amount of events PER category for training', int
        self.dataset['validAmount'] = 2_500, 'the amount of events PER category for validation', int
        self.dataset['normalize'] = True, 'wether or not to normalize data on import'

        self.tree = Section('tree')
        self.tree['depth'] = 5, "how deep the tree can be", int
        self.tree['minSamples'] = 2, 'the minimum sample split size', int
        self.tree['impurity'] = 'gini', "the impurity measure function", str, ['gini', 'entropy', 'mae', 'mse']
        self.tree['leaf'] = 'mode', "how leaf values are set", str, ['mode', 'mean']
        self.tree['split'] = 'id3', "the split algorithm", str, ['id3', 'c45', 'cart']
        self.tree['percentile'] = None, "the percentile of feature values to access during training", int
        self.tree['featSelection'] = None, "a preselection of features", [None, str], [None, 'variance', 'random', 'mutual', 'kendall', 'anova', 'choice']
        self.tree['featParameter'] = None, "a threshold feautres need to pass or a number of features to be picked", [None, float, int, list]

        super().__init__()