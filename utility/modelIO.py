from typing import Protocol, runtime_checkable
import json
from zipfile import ZipFile, ZIP_DEFLATED
from importlib import import_module
from datetime import datetime


@runtime_checkable
class Saveable(Protocol):
    """
    this protocol should be implemented in every class
    that should be save/load-able from a dict.
    keep in mind, that only native python dtypes work wiht json
    """
    @property
    def qualifiedName(self) -> tuple:
        """
        module name and class name
        """
        ...

    def toDict(self) -> dict:
        """
        converts class parameters to a python dict
        all parameters need to be in native python format
        """
        pass

    @classmethod
    def fromDict(cls: object, loadDict: dict) -> object:
        """
        loads class parameters from a dict
        """
        pass


class ModelIO(object):
    @classmethod
    def load(cls, filename: str) -> object:
        # adding the file name extension
        if not filename.endswith('.modelzip'):
            filename += '.modelzip'

        # Loading the zip file of the model
        with ZipFile(filename, 'r') as zipArchieve:
            # reading the json file in the zip archive
            with zipArchieve.open(zipArchieve.namelist()[0]) as f:
                loadDict = json.load(f)

        # removing unnecessary data for object recreation
        moduleName, className = loadDict.pop('qualifiedName')  # assuming qualifiedName exists in the dict
        loadDict.pop('datetime')

        # loading the proper modules
        Module = import_module(moduleName)  # dynamically import module
        Class = getattr(Module, className)  # get class from imported module

        try: # recreate the object from dict
            return Class.fromDict(loadDict)
        except AttributeError:
            raise TypeError(f'{className} is not Saveable.')

    @classmethod
    def save(cls, object: object, filename: str) -> None:
        # checking if object is savable
        if not isinstance(object, Saveable):
            raise TypeError(f'{object.__class__.__name__} is not Saveable.')

        # converting the obejct to a dict
        objectDict = object.toDict()

        # setting up the save dict
        saveDict = {}
        saveDict['datetime'] = datetime.now().isoformat()
        saveDict['qualifiedName'] = object.qualifiedName

        # adding the object dict to savedict, this keeps the date at the beginning
        # the order is not important, it's just to make the date more easily findable
        saveDict.update(objectDict)

        # adding the file name extension
        if not filename.endswith('.modelzip'):
            filename += '.modelzip'

        # saving the file
        with ZipFile(filename, 'w', ZIP_DEFLATED) as zipArchieve:
            jsonString = json.dumps(saveDict, indent=4)
            with zipArchieve.open('model.json', 'w') as f:
                f.write(jsonString.encode('utf-8'))
