from typing import Any
import numpy as np
import os, re
from ast import literal_eval


def getType(input: str):
    """
    this function returns a variable as the type it is meant to be used
    I am using it to assess variables from config files
    """
    # first this function tries to guess a data type, it will fail in cases of strings
    try:
        tupleOpen = input.count('(')
        tupleClose = input.count(')')
        if tupleOpen == 1 and tupleClose == 1:
            return literal_eval(input)
        input = literal_eval(input)
        # here tuples will be converted into lists, simply because literal_eval seems to prefer tuples over lists, I don't
        if type(input) == tuple:
            return list(input)
        else:
            return input
    # here everything that has been deemed to be a string will be parsed, either they are strings or strings in lists
    except (ValueError, SyntaxError):
        input = input.replace(' ','').replace(';', ',') # getting rid of semi colons and spaces
        # this is is the user writes a list with brackets, a bit more of work is needed to turn it into potentially nested lists
        if '[' in input:
            input = input.replace('[','')
            input = input.split('],')
            output = []
            for line in input:
                line = line.replace(']','')
                output.append(line.split(','))
            return output
        # this just splits a list of strings into a list full of strings
        elif ',' in input:
            return input.split(',')
        # just returns the string as is
        else:
            return input


def wagnerFischer(Q: str, R: str) -> int:
    """
    finding the edit distance between two strings
    """
    matrix = np.zeros((len(Q) + 1, len(R) + 1))
    for i in range(len(Q) + 1):
        matrix[i, 0] = i
    for j in range(len(R) + 1):
        matrix[0, j] = j
    for i in range(1, len(Q) + 1):
        for j in range(1, len(R) + 1):
            x = matrix[i - 1, j] + 1
            y = matrix[i, j - 1] + 1
            z = matrix[i - 1, j - 1]
            if Q[i - 1] != R[j - 1]:
                z += 1
            matrix[i, j] = min(x, y, z)
    return int((matrix[len(Q), len(R)]))


def flatten(List: list) -> list:
    """
    this function flattens python lists, since they can be ragged and numpy/pytorch have troubles handling them
    this is used in two places in the code, here and in dataclass
    """
    returnList = []
    for item in List:
        if type(item) == list:
            elements = flatten(item)
            for element in elements:
                returnList.append(element)
        else:
            returnList.append(item)
    return returnList


class Setting(object):
    """
    this class handles a SINGLE parameter of the network, it takes care of options and types, so that nothing gets assigned incorrectly
    """
    def __init__(self, name: str, default, tooltip: str = '', types: list = [None], options: list = [None], fallback: Any = None) -> None:
        # name and tooltip of this setting
        self.name = name
        self.tooltip = tooltip

        # the value stored in this setting, the default value and a fallback
        self.value = default
        self.default = default
        self.fallback = default if fallback is None else fallback

        # assigning the types this class can handle
        if types == [None]:
            self.types = [type(default)]
        else:
            self.types = flatten([types])

        # if the assigned value should be in a list
        if len(self.types) > 1 and list in self.types:
            self.convertToList = True
        else:
            self.convertToList = False

        # the possible options of this setting
        self.options = options
        if type(default) == bool and options == [None]:
            self.options = [True, False]

    def checks(self, value: Any) -> tuple[bool, bool]:
        """
        this checks if an input makes sense for this parameter, no assigning is happening here
        """
        typeBool, optionBool = False, False
        if type(value) in self.types:
            typeBool = True
        if isinstance(value, list) and not self.options == [None]:
            value = flatten(value)
            checks = [False]*len(value)
            for i,item in enumerate(value):
                checks[i] = item in self.options
            optionBool = all(checks)
        elif value in self.options or self.options == [None]:
            optionBool = True
        return typeBool, optionBool

    def help(self) -> tuple:
        """
        this just return some information about this setting/parameter
        """
        return self.name, self.types, self.options, self.tooltip

    def __repr__(self) -> str:
        return str(self.value)

    def __len__(self) -> int:
        return len(self.value)


class Section(object):
    """
    this class finally contains parameters, but only of single section, it handels the creation and setting of the actual parameters
    the networks/programms parameter are variables/parameters of this class, which can be called like normal values/variables
    it also stores all incorrectly user parameters
    it handles the printing and writing of its own section parameters
    """
    def __init__(self, section: str, *, controlParam: str | None = None, completionMode: int = 0, identify: bool = False) -> None:
        # name and metrics of this settings section
        self.__sectionName__ = section
        self.__maxLen__ = 0
        self.__wrong__ = {}
        self.__numAttr__ = 0

        # if to check consistency of parameters
        self.__controlParam__ = controlParam
        self.__completionMode__ = completionMode

        # this is used when parameters are stored in dicts
        self.__identify__ = identify

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key).value

    def __setitem__(self, key: str, value: Any) -> None:
        """
        this sets/assigns parameters to this class, calls functions of the Setting class inorder to determine if a user input can be assigned
        """
        if key not in self.list():
            self.__numAttr__ += 1
            if type(value) == tuple:
                setattr(self, key, Setting(key, *value))
            else:
                setattr(self, key, Setting(key, value))
        else:
            typesCheck, optionsCheck = getattr(self, key).checks(value)
            if getattr(self, key).convertToList is True and not type(value) == list:
                value = [value]
            if all((typesCheck, optionsCheck)):
                getattr(self, key).value = value
            elif typesCheck is False:
                self.__wrong__[key] = [type(value), 'Wrong Type', getattr(self, key).types]
            elif typesCheck is True and optionsCheck is False:
                if not getattr(self, key).options == [None]:
                    self.__wrong__[key] = [type(value), 'No Option', getattr(self, key).options]
                else:
                    self.__wrong__[key] = [type(value), 'No Option', getattr(self, key).default]
        if self.__maxLen__ < len(key):
            self.__maxLen__ = len(key)

    def setConfig(self, config: dict) -> None:
        """
        setting the cofiguration
        """
        self.__config__ = config

        # wether to check constancy
        if self.__controlParam__ is not None and self.__identify__ is False:
            self._consistencyCheck()

        # constancy of parameters stored in dicts?
        elif self.__identify__ is True:
            self._identify()

        # setting the config
        for key in self.__config__:
            if key in self.list():
                self[key] = self.__config__[key]


    def _paramLength(self, param: list, length: int) -> list:
        """
        changes the length of a config parameter
        """
        if type(param) is not list:
            return param

        # completion mode sets wether to have some length or different
        diffLen = length - len(param) + self.__completionMode__
        if diffLen > 0:
            extensionList = [param[-1]] * diffLen
            param.extend(extensionList)
        elif diffLen < 0:
            param = param[:diffLen]
        return param

    def _consistencyCheck(self) -> None:
        """
        this checks the consistency of all settings in this section
        parameters are stored in lists
        """
        if self.__controlParam__ not in self.__config__:
            paramLen = max([len(self.__config__[key]) for key in self.__config__ if hasattr(self, key) and type(self.__config__[key]) == list], default=0)
            self.__config__[self.__controlParam__] = paramLen - self.__completionMode__

        paramLen = self.__config__[self.__controlParam__]
        for key in self.list():
            if key is self.__controlParam__:
                continue

            if key not in self.__config__:
                self.__config__[key] = getattr(self, key).fallback

            typesCheck, optionsCheck = getattr(self, key).checks(self.__config__[key])
            if typesCheck is False or optionsCheck is False:
                self.__config__[key] = getattr(self, key).fallback

            if type(self.__config__[key]) is not list:
                self.__config__[key] = [self.__config__[key]]

            self.__config__[key] = self._paramLength(self.__config__[key], paramLen)

    def _identify(self) -> None:
        """
        this does the same as '_consistencyCheck', but for settings
        stored in dicts, which happens when the user needs the same,
        but different setting for multiply objects
        """
        lengthes = {}
        paramIDs = []
        keys = list(self.__config__.keys())
        for key in keys:
            paramID = re.findall(r'[0-9]+', key)

            if len(paramID) > 0:
                variable = key.replace(paramID[0], '')
                paramID = int(paramID[0])
            else:
                variable = key
                paramID = 0

            if paramID not in paramIDs:
                paramIDs.append(paramID)

            if variable == self.__controlParam__:
                if paramID in lengthes:
                    lengthes[paramID] = max(lengthes[paramID], self.__config__[key])
                else:
                    lengthes[paramID] = self.__config__[key]
            else:
                if paramID in lengthes:
                    lengthes[paramID] = max(lengthes[paramID], len(self.__config__[key]) - self.__completionMode__)
                else:
                    lengthes[paramID] = len(self.__config__[key]) - self.__completionMode__
                if type(self.__config__[key]) is not list:
                    self.__config__[key] = [self.__config__[key]]

            if variable not in self.__config__:
                self.__config__[variable] = {}
                self.__config__[variable][paramID] = self.__config__[key]
            elif type(self.__config__[variable]) is not dict:
                cached = self.__config__[variable]
                self.__config__[variable] = {}
                self.__config__[variable][paramID] = cached
            else:
                self.__config__[variable][paramID] = self.__config__[key]

            if paramID != 0:
                self.__config__.pop(key)

        for key in self.list():
            for id in paramIDs:
                if key not in self.__config__:
                    self.__config__[key] = {}
                if id not in self.__config__[key]:
                    self.__config__[key][id] = getattr(self, key).fallback

                if key == self.__controlParam__:
                    self.__config__[key][id] = lengthes[id]
                self.__config__[key][id] = self._paramLength(self.__config__[key][id], lengthes[id])

    def __repr__(self) -> str:
        printString = (' ' + self.__sectionName__ + ' ').center(55,'—') + '\n'
        for i, key in enumerate(self.list()):
            if i == self.__numAttr__-1:
                printString += "{} = {}".format(str(key).ljust(self.__maxLen__), getattr(self, key))
            else:
                printString += "{} = {}\n".format(str(key).ljust(self.__maxLen__), getattr(self, key))
        return printString

    def list(self):
        """
        this lists all attributes of the class
        """
        return [s for s in vars(self).keys() if not (s.startswith('__') or callable(getattr(self, s)))]


class Settings(object):
    """
    base class for settings
    """
    def __init__(self) -> None:
        self.__searchDict__ = {}
        for item in self.list():
            for element in getattr(self, item).list():
                self.__searchDict__[element] = item

        self.__config__ = {} # this stores the content of a config file

        self.__wrong__ = [] # stores incorrect setting names
        self.__maybe__ = [] # list of what the user possibly meant

        # contains rejected settings and the reason of their rejection
        # this could be type or wrong choice, not an option
        self.__rejected__ = {}

    def __getitem__(self, key: str) -> Any:
        try:
            masterKey = self.__searchDict__[key]
            return getattr(self, masterKey)[key]
        except KeyError:
            print('{} is not an attribute of settingsClass'.format(key))

    def __setitem__(self, key: str, value: Any) -> Any:
        masterKey = self.__searchDict__[key]
        getattr(self, masterKey)[key] = value

    def __repr__(self) -> str:
        printString = ' settings '.center(55, '━') + '\n'
        for item in self.list():
            printString += str(getattr(self, item)) + '\n'

        if len(self.__rejected__) > 0:
            header = " wrong data type or option "
            header = header.center(55,'—')
            printString += "{}\n".format(header)
            maxLen = np.vectorize(len)(list(self.__rejected__.keys()))[0]
            headding = ''.center(maxLen) + '  ' + 'error'.center(15) + '  ' + 'options'.center(15)
            printString += headding + '\n'
            for key in self.__rejected__:
                string1, string2 = str(self.__rejected__[key][1]), str(self.__rejected__[key][2]).replace('<class ','').replace('>','')
                printString += key.rjust(maxLen) + '  ' + string1.center(15) + '  ' + string2.center(15) + '\n'
        if len(self.__wrong__) > 0:
            if len(self.__maybe__) == 0:
                self.__didYouMean__()
            header = " did you mean? "
            header = header.center(55,'—')
            printString += "{}\n".format(header)
            maxLen = np.vectorize(len)(self.__wrong__).max()
            for item, maybe in zip(self.__wrong__,self.__maybe__):
                printString += str(item).rjust(maxLen) + ' => ' + str(maybe) + '\n'

        return printString + '\n'

    def getConfig(self, configFile: str | None = None) -> None:
        if configFile is None:
            print('no config specified, continuing with defualts')
            return
        #if os.path.isfile(configFile):
        #f = open(configFile, 'r').read().split('\n')
        with open(configFile, 'r') as f:
            for line in f.read().split('\n'):
                line = line.replace(' ', '')
                line, _, _ = line.partition('#')
                if len(line) == 0:
                    continue
                line = line.replace(':', '=').split('=')

                paramID = re.findall(r'[0-9]+', line[0])
                if len(paramID) > 0:
                    variable = line[0].replace(paramID[0], '')
                else:
                    variable = line[0]

                if variable in (self.__searchDict__.keys()):
                    key = self.__searchDict__[variable]
                    if line[1] == 'true' or line[1] == 'false':
                        line[1] = line[1].capitalize()
                    if key not in self.__config__:
                        self.__config__[key] = {}
                    self.__config__[key][line[0]] = getType(line[1])
                else:
                    self.__wrong__.append(line[0])

            print("config file '{}' found and read".format(configFile))
        #else:
        #    print("config file '{}' not found, continuing with defualts".format(configFile))

    def setConfig(self) -> None:
        if len(self.__config__) > 0:
            print('set config....')
            for key in self.__config__:
                getattr(self, key).setConfig(self.__config__[key])

            # here I collect all parameter configs together that were
            # rejected for being wrong type or an invalid option
            # this '|' operator merges dicts
            for item in self.list():
                self.__rejected__ = self.__rejected__ | getattr(self, item).__wrong__

    def __didYouMean__(self):
        dist = np.zeros((len(self.__wrong__),len(list(self.__searchDict__.keys()))))
        self.__maybe__ = ['']*len(self.__wrong__)
        for i,element in enumerate(self.__wrong__):
            for j,item in enumerate(list(self.__searchDict__.keys())):
                dist[i][j] = wagnerFischer(element,item)
            self.__maybe__[i] = list(self.__searchDict__.keys())[dist[i].argmin()]

    def list(self) -> list:
        return [s for s in vars(self).keys() if not (s.startswith('__') or callable(getattr(self, s)))]
