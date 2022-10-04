from typing import Union
from numpy import ndarray
from torch import tensor

class DirichletPoints():
    def __init__(self, 
        x:Union[ndarray, tensor],
        val:Union[ndarray, tensor],
        ) -> None:
        assert x.shape[0] == val.shape[0]
        self.x = x
        self.val = val
        
class NeumannPoints():
    def __init__(self, 
        x:Union[ndarray, tensor],
        val:Union[ndarray, tensor],
        normal:Union[ndarray, tensor],
        ) -> None:
        assert x.shape[0] == val.shape[0]
        assert x.shape == normal.shape
        self.x = x
        self.val = val
        self.normal = normal

class InternalPoints():
    def __init__(self, 
        x:Union[ndarray, tensor],
        source:Union[ndarray, tensor],
        ) -> None:
        assert x.shape[0] == source.shape[0]
        self.x = x
        self.source = source

class InternalPointsSolidMixedForm(InternalPoints):
    def __init__(self, 
        x:Union[ndarray, tensor],
        source:Union[ndarray, tensor],
        disp:Union[ndarray, tensor],
        strain:Union[ndarray, tensor],
        ) -> None:
        super().__init__(x, source)
        assert x.shape[0] == disp.shape[0]
        assert x.shape[0] == strain.shape[0]
        if x.shape[1] == 2:
            assert strain.shape[1] == 3
        else:
            raise NotImplementedError(strain.shape[1])
        self.disp = disp
        self.strain = strain
