import numpy as np
from numpy.typing import NDArray
import warnings
warnings.filterwarnings("error")


class ConfusionMatrix:
    """
    This class creates a confusion matrix based on labels
    It calculates also the performance scores based on the confusion matrix
    One can update the scores to be calculated
    """
    __slots__ = ['name', 'numClasses', 'matrix', 'procent', 'classes', 'classNames', 'nameLength', 'scoreNames', 'scoreLength', 'scoreFormular', 'scores', 'totals', '_scoreByFormular', '_socreByName', '_wrongFormular', '_wrongName']

    def __init__(self, numClasses: int | None = None, classNames: list[str] | None = None) -> None:
        self.name = self.__class__.__name__

        if numClasses is None and classNames is None:
            raise ValueError('need to give either/both number of classes or class names')
        if numClasses is not None and classNames is not None:
            if len(classNames) != numClasses:
                raise ValueError('number of classes must be length of class names')

        # matrix
        self.numClasses = len(classNames) if numClasses is None else numClasses
        self.matrix = np.zeros((self.numClasses, self.numClasses), dtype=int)
        self.procent = np.zeros((self.numClasses, self.numClasses), dtype=float)
        self.classes = np.arange(0, self.numClasses)

        # class names
        self.classNames = [f'Class {i}' for i in range(self.numClasses)] if classNames is None else classNames
        self.nameLength = [len(item) for item in self.classNames]

        # scores
        self.scoreNames = ['accuracy', 'precision', 'sensitivity', 'miss rate']
        self.scoreLength = [len(item) for item in self.scoreNames]
        self.scoreFormular = ['(tp+tn)/(tp+tn+fp+fn)', 'tp/(tp+fp)', 'tp/(tp+fn)', 'fn/(fn+tp)']
        self.scores = np.zeros((len(self.classNames), len(self.scoreNames)))

        # total scores
        self.totals = np.zeros(len(self.scoreNames))

        # all possible score formulars and score names, used for configuring scores by the user
        self._scoreByFormular = {'tp/(tp+fn)': 'sensitivity',
                                 'tp/(fn+tp)': 'sensitivity',

                                 'tn/(tn+fp)': 'rejection',
                                 'tn/(fp+tn)': 'rejection',

                                 'fn/(fn+tp)': 'miss rate',
                                 'fn/(tp+fn)': 'miss rate',

                                 'tp/(tp+fp)': 'precision',
                                 'tp/(fp+tp)': 'precision',

                                 'fp/(fp+tn)': 'fall-out',
                                 'fp/(tn+fp)': 'fall-out',

                                 'fn/(fn+tn)': 'false omission',
                                 'fn/(tn+fn)': 'false omission',

                                 'fp/(fp+tp)': 'false dicovery',
                                 'fp/(tp+fp)': 'false dicovery',

                        '(2*tp)/(2*tp+fp+fn)': 'f1 score',
                        '(2*tp)/(2*tp+fn+fp)': 'f1 score',
                        '(2*tp)/(fn+2*tp+fp)': 'f1 score',
                        '(2*tp)/(fp+fn+2*tp)': 'f1 score',
                        '(2*tp)/(fn+fp+2*tp)': 'f1 score',
                        '(2*tp)/(fp+2*tp+fn)': 'f1 score',

                              'tp/(tp+fn+fp)': 'threat score',
                              'tp/(tp+fp+fn)': 'threat score',
                              'tp/(fp+tp+fn)': 'threat score',
                              'tp/(fn+fp+tp)': 'threat score',
                              'tp/(fp+fn+tp)': 'threat score',
                              'tp/(fn+tp+fp)': 'threat score',

                      '(tp+tn)/(tp+fn+fp+tn)': 'accuracy',
                      '(tp+tn)/(tp+fn+tn+fp)': 'accuracy',
                      '(tp+tn)/(tp+fp+fn+tn)': 'accuracy',
                      '(tp+tn)/(tp+fp+tn+fn)': 'accuracy',
                      '(tp+tn)/(tp+tn+fn+fp)': 'accuracy',
                      '(tp+tn)/(tp+tn+fp+fn)': 'accuracy',
                      '(tp+tn)/(fn+tp+fp+tn)': 'accuracy',
                      '(tp+tn)/(fn+tp+tn+fp)': 'accuracy',
                      '(tp+tn)/(fn+fp+tp+tn)': 'accuracy',
                      '(tp+tn)/(fn+fp+tn+tp)': 'accuracy',
                      '(tp+tn)/(fn+tn+tp+fp)': 'accuracy',
                      '(tp+tn)/(fn+tn+fp+tp)': 'accuracy',
                      '(tp+tn)/(fp+tp+fn+tn)': 'accuracy',
                      '(tp+tn)/(fp+tp+tn+fn)': 'accuracy',
                      '(tp+tn)/(fp+fn+tp+tn)': 'accuracy',
                      '(tp+tn)/(fp+fn+tn+tp)': 'accuracy',
                      '(tp+tn)/(fp+tn+tp+fn)': 'accuracy',
                      '(tp+tn)/(fp+tn+fn+tp)': 'accuracy',
                      '(tp+tn)/(tn+tp+fn+fp)': 'accuracy',
                      '(tp+tn)/(tn+tp+fp+fn)': 'accuracy',
                      '(tp+tn)/(tn+fn+tp+fp)': 'accuracy',
                      '(tp+tn)/(tn+fn+fp+tp)': 'accuracy',
                      '(tp+tn)/(tn+fp+tp+fn)': 'accuracy',
                      '(tp+tn)/(tn+fp+fn+tp)': 'accuracy',
                      '(tn+tp)/(tp+fn+fp+tn)': 'accuracy',
                      '(tn+tp)/(tp+fn+tn+fp)': 'accuracy',
                      '(tn+tp)/(tp+fp+fn+tn)': 'accuracy',
                      '(tn+tp)/(tp+fp+tn+fn)': 'accuracy',
                      '(tn+tp)/(tp+tn+fn+fp)': 'accuracy',
                      '(tn+tp)/(tp+tn+fp+fn)': 'accuracy',
                      '(tn+tp)/(fn+tp+fp+tn)': 'accuracy',
                      '(tn+tp)/(fn+tp+tn+fp)': 'accuracy',
                      '(tn+tp)/(fn+fp+tp+tn)': 'accuracy',
                      '(tn+tp)/(fn+fp+tn+tp)': 'accuracy',
                      '(tn+tp)/(fn+tn+tp+fp)': 'accuracy',
                      '(tn+tp)/(fn+tn+fp+tp)': 'accuracy',
                      '(tn+tp)/(fp+tp+fn+tn)': 'accuracy',
                      '(tn+tp)/(fp+tp+tn+fn)': 'accuracy',
                      '(tn+tp)/(fp+fn+tp+tn)': 'accuracy',
                      '(tn+tp)/(fp+fn+tn+tp)': 'accuracy',
                      '(tn+tp)/(fp+tn+tp+fn)': 'accuracy',
                      '(tn+tp)/(fp+tn+fn+tp)': 'accuracy',
                      '(tn+tp)/(tn+tp+fn+fp)': 'accuracy',
                      '(tn+tp)/(tn+tp+fp+fn)': 'accuracy',
                      '(tn+tp)/(tn+fn+tp+fp)': 'accuracy',
                      '(tn+tp)/(tn+fn+fp+tp)': 'accuracy',
                      '(tn+tp)/(tn+fp+tp+fn)': 'accuracy',
                      '(tn+tp)/(tn+fp+fn+tp)': 'accuracy'}

        self._socreByName = {'sensitivity': 'tp/(tp+fn)',
                                  'recall': 'tp/(tp+fn)',
                                 'hitrate': 'tp/(tp+fn)',
                        'truepositiverate': 'tp/(tp+fn)',
                            'truepositive': 'tp/(tp+fn)',
                                     'tpr': 'tp/(tp+fn)',

                             'specificity': 'tn/(tn+fp)',
                             'selectivity': 'tn/(tn+fp)',
                        'truenegativerate': 'tn/(tn+fp)',
                            'truenegative': 'tn/(tn+fp)',
                                     'tnr': 'tn/(tn+fp)',

                               'precision': 'tp/(tp+fp)',
                 'positivepredictivevalue': 'tp/(tp+fp)',
                      'positivepredictive': 'tp/(tp+fp)',
                                     'ppv': 'tp/(tp+fp)',

                               'rejection': 'tn/(tn+fn)',
                 'negativepredictivevalue': 'tn/(tn+fn)',
                      'negativepredictive': 'tn/(tn+fn)',
                                     'npv': 'tn/(tn+fn)',

                                'missrate': 'fn/(fn+tp)',
                       'falsenegativerate': 'fn/(fn+tp)',
                           'falsenegative': 'fn/(fn+tp)',
                                     'fnr': 'fn/(fn+tp)',

                                 'fallout': 'fp/(fp+tn)',
                       'falsepositiverate': 'fp/(fp+tn)',
                           'falsepositive': 'fp/(fp+tn)',
                                     'fpr': 'fp/(fp+tn)',

                      'falsediscoveryrate': 'fp/(fp+tp)',
                          'falsediscovery': 'fp/(fp+tp)',
                                     'fdr': 'fp/(fp+tp)',

                       'falseomissionrate': 'fn/(fn+tn)',
                           'falseomission': 'fn/(fn+tn)',
                                     'for': 'fn/(fn+tn)',

                             'threatscore': 'tp/(tp+fn+fp)',
                    'criticalsuccessindex': 'tp/(tp+fn+fp)',
                         'criticalsuccess': 'tp/(tp+fn+fp)',
                                      'ts': 'tp/(tp+fn+fp)',
                                     'csi': 'tp/(tp+fn+fp)',

                                'accuracy': '(tp+tn)/(tp+fn+fp+tn)',
                                     'acc': '(tp+tn)/(tp+fn+fp+tn)',

                                 'f1score': '(2*tp)/(2*tp+fp+fn)'}
        self._wrongFormular = []
        self._wrongName = []

    def update(self, prediction: NDArray, target: NDArray) -> None:
        """
        Update the confusion matrix based on new predictions and targets.
        """

        # convert one-hot encoding to categorial
        if len(target.shape) == 2:
            target = np.argmax(target, axis=-1)
            prediction = np.argmax(prediction, axis=-1)

        # loop across the different combinations of actual / predicted classes
        for i in range(self.numClasses):
            for j in range(self.numClasses):
                # count the number of instances in each combination of actual / predicted classes
                self.matrix[i, j] += np.sum((target == self.classes[i]) & (prediction == self.classes[j]))

    def percentages(self) -> None:
        # Convert the confusion matrix to percentages.
        self.procent = np.round(100 * (self.matrix / self.matrix.sum()), 2)

    def calcScores(self) -> None:
        """
        Calculate the scores based on the confusion matrix.
        """

        # reading tp, tn, fp, tn from confusion matrix
        tptensor = self.matrix.diagonal()
        fntensor = (self.matrix.sum(1) - self.matrix.diagonal())#.reshape(-1,1)
        fptensor = (self.matrix.sum(0) - self.matrix.diagonal())#.reshape(-1,1)
        tntensor = (self.matrix.sum() - self.matrix.sum(1) - self.matrix.sum(0) + self.matrix.diagonal())#.reshape(-1,1)

        # calculating scores
        for i, formular in enumerate(self.scoreFormular):
            for j, category in enumerate(self.classNames):
                # tp, fn, fp, tn will be used by 'eval(formular)'
                # every formular is a string consiting of tp, tn, fp, tn
                # pyflakes says these variables are never used, but
                # they are used with 'formular', where eval(...) converts
                # a string into code, which uses tp, fn, fp and tn
                tp, fn, fp, tn = tptensor[j], fntensor[j], fptensor[j], tntensor[j]
                try:
                    self.scores[j,i] = eval(formular)
                except RuntimeWarning:
                    self.scores[j,i] = np.nan

        # estimating the total scores across all categories
        self.totals = np.nanmean(self.scores, axis=0)
        #self.totals = self.scores.mean(0)

    def setScores(self, *scores: str) -> None:
        """
        allows setting custom scores, needs to be based confusion matrix
        'scores' either needs to be a list of names of formulars
        when call, it overwrites all scores
        """
        self.scoreNames = []
        self.scoreFormular = []
        for score in scores:
            name = score.lower().replace(' ','').replace('-','')
            if score in self._scoreByFormular:
                self.scoreNames.append(self._scoreByFormular[score])
                self.scoreFormular.append(score)
            elif name in self._socreByName:
                self.scoreNames.append(score)
                self.scoreFormular.append(self._socreByName[name])
            elif '(' in score or ')' in score or '+' in score or '-' in score or '/' in score or '*' in score:
                self._wrongFormular.append(score)
            else:
                self._wrongName.append(score)

    def __str__(self) -> str:
        """
        Print the confusion matrix and the scores.
        """
        lengthAddition = 5
        center = (self.numClasses + 1) * (np.max(self.nameLength) + lengthAddition)
        printString = ''

        # printing the section title
        if np.sum(self.scores) > 0:
            printString += ' evaluation '.center(center, '━') + '\n'

        # printing the confusion matrix
        printString += ' confusion matrix '.center(center, '—') + '\n'
        printString += ''.ljust(np.max(self.nameLength) + lengthAddition)
        for head in self.classNames:
            printString += head.center(np.max(self.nameLength) + lengthAddition)
        printString += '\n' + '·' * (center) + '\n'
        for i, (line, pro) in enumerate(zip(self.matrix, self.procent)):
            printString += self.classNames[i].rjust(np.max(self.nameLength) + lengthAddition)
            for item in line:
                printString += str(int(item.item())).center(np.max(self.nameLength) + lengthAddition)
            printString += '\n'
            if np.sum(self.procent) > 0:
                printString += ''.rjust(np.max(self.nameLength) + lengthAddition)
                for item in pro:
                    printString += (str(int(item.item())) + '%').center(np.max(self.nameLength) + lengthAddition)
                printString += '\n'
                if i < self.numClasses - 1:
                    printString += '·' * (center) + '\n'

        # printing the scores
        if np.sum(self.scores) > 0:
            center = np.max(self.nameLength) + len(self.scoreNames) * (np.max(self.scoreLength) + lengthAddition)
            printString += '\n' + ' scores '.center(center, '—') + '\n'
            printString += ''.ljust(np.max(self.nameLength) + lengthAddition)
            for head in self.scoreNames:
                printString += head.center(np.max(self.scoreLength) + lengthAddition)
            printString += '\n' + '·' * (center) + '\n'
            for i, line in enumerate(self.scores):
                printString += self.classNames[i].rjust(np.max(self.nameLength) + lengthAddition)
                for item in line:
                    printString += str(round(item.item(),3)).center(np.max(self.scoreLength) + lengthAddition)
                printString += '\n'
            printString += '·' * (center) + '\n'
            printString += 'total'.rjust(np.max(self.nameLength) + lengthAddition)
            for item in self.totals:
                printString += str(round(item.item(),3)).center(np.max(self.scoreLength) + lengthAddition)
        return printString
