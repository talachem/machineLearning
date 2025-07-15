import numpy as np
from .format import Format


class Progressbar():
    """
    this creates a progress-bar for different processes, I use it during training/validation
    """
    __slots__ = ['ff', 'colors', 'printLength', 'symbols', 'subLength', 'totalLength', 'length', 'elements', 'steps', 'subLengthes', 'title', 'partOne', 'timeStep']

    def __init__(self, title: str, length: int, printLength: int = 49):
        self.ff = Format()
        self.colors = ['red', 'yellow', 'green', 'white']
        self.printLength = printLength
        self.symbols = [' ', '⠆', '⡇', '⡷', '⣿']
        self.subLength = len(self.symbols)
        self.totalLength = self.printLength * self.subLength
        self.length = int(length)
        self.elements = np.vectorize(int)(np.arange(0,self.printLength+1,(self.printLength+1)/self.length))
        # the bar steps through the symbols listed above, sadly I didn't find a better way of creating this stepping effect, but with a lot fuzz fuzz
        self.steps = np.array([], dtype=int)
        self.subLengthes = np.unique(self.elements,return_counts=True)[1]
        for item in self.subLengthes:
            self.steps = np.append(self.steps, np.vectorize(int)(np.arange(0,self.subLength,self.subLength/item)))
        self.title = title
        self.timeStep = 0
        self.partOne = self.title + ' |'
        partTwo = ' ' * (self.printLength + 1)
        partFive = '|' + ' 00%'
        print(self.partOne + partTwo + partFive, end='\r')

    def step(self):
        """
        this steps through the bar, should be called within the loop
        """
        if self.timeStep < self.length - 1:
            length = self.elements[self.timeStep]
            percent = (self.timeStep/self.length)
            index = int(percent * (len(self.colors) - 1))
            partTwo = self.ff(self.symbols[-1] * length, color=self.colors[index])
            partThree = self.ff(self.symbols[self.steps[self.timeStep]], color=self.colors[index])
            partFour = ' ' * (self.printLength - length)
            partFive = '| ' + str(int(percent*100)).zfill(2) + '%'
            toPrint = self.partOne + partTwo + partThree + partFour + partFive
            endChar = '\r'
        else:
            partTwo = self.symbols[-1] * (self.printLength + 1) + '|'
            partFive = ' ' + 'done ✔'
            toPrint = self.partOne + partTwo + partFive
            endChar = '\n'
        self.timeStep += 1
        print(toPrint, end=endChar, flush=True)

    def finish(self):
        partTwo = self.symbols[-1] * (self.printLength + 1) + '|'
        partFive = ' ' + 'done ✔'
        toPrint = self.partOne + partTwo + partFive
        endChar = '\n'
        print(toPrint, end=endChar, flush=True)

    def stepTo(self, timeStep):
        length = self.elements[timeStep]
        percent = (timeStep/self.length)
        index = int(percent * (len(self.colors) - 1))

        partTwo = self.ff(self.symbols[-1] * length, color=self.colors[index])
        partThree = self.ff(self.symbols[self.steps[self.timeStep]], color=self.colors[index])
        partFour = ' ' * (self.printLength - length)
        partFive = '| ' + str(int(percent*100)).zfill(2) + '%'

        toPrint = self.partOne + partTwo + partThree + partFour + partFive
        endChar = '\r'

        print(toPrint, end=endChar, flush=True)