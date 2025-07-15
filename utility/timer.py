import numpy as np
import time


def printTime(time: int) -> str:
    """
    converts seconds into a legible time expression
    """
    hours = int(np.floor(time/(60*60)))
    if hours >= 1:
        time = time % (60*60)
    minutes = int(np.floor(time/60))
    seconds = time % 60
    if hours >= 1 and minutes > 0:
        return f"{hours} h, {minutes} min & {round(seconds,1)} sec"
    elif hours >= 1 and minutes == 0:
        return f"{hours} h, {round(seconds,1)} sec"
    elif minutes >= 1:
        return f"{minutes} min, {round(seconds,1)} sec"
    else:
        return f"{round(seconds,1)} sec"


def dotAligned(seq: list) -> [str]:
    """
    aligns a bunch of floats to the decimal dot
    """
    snums = [str(n) for n in seq]
    dots = []
    for s in snums:
        p = s.find('.')
        if p == -1:
            p = len(s)
        dots.append(p)
    m = max(dots)
    return [' ' * (m - d) + s for s, d in zip(snums, dots)]


class Time():
    """
    a class for tracking time
    """
    __slots__ = ['times', '__start__']

    def __init__(self):
        self.times = {'total time': 0}
        self.__start__ = 0

    def start(self):
        """
        start tracking time
        """
        self.__start__ = time.time()

    def record(self, name):
        """
        'total time' tracks automatically all times
        """
        if name in self.times:
            self.times[name] += (time.time() - self.__start__)
        else:
            self.times[name] = time.time() - self.__start__
        self.times['total time'] += time.time() - self.__start__

    def __str__(self):
        """
        this function prints/writes out the recorded times, if no title is provided it prints to cli
        """
        center = 60
        maxLen = np.vectorize(len)(list(self.times.keys())).max()
        printString = " times ".center(center, '—') + '\n'

        self.times['total time'] = self.times.pop('total time') # this is done in order to put total time at the end of the list
        printTimes = np.vectorize(printTime)(list(self.times.values()))
        printValues = dotAligned(list(self.times.values()))
        timeLen = np.vectorize(len)(printTimes).max()

        # iterating over times
        for key, times, printValue in zip(self.times, printTimes, printValues):
            word = key.rjust(maxLen) + '   '
            if key == 'total time':
                printString += '·'*center + '\n'
            printString += word + times.ljust(timeLen) + '    ' + printValue + '\n'
        return printString