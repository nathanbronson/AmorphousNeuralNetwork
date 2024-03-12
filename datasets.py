from __future__ import annotations

from typing import Any, List, Optional, Tuple
from itertools import combinations_with_replacement as combos

import numpy as np

from utils import one_hot_int

class Dataset(object):
    """
    Base Dataset Class

    Attributes (name: type = use):
        x: List[List[Optional[float]]] = all input data of
                                         dimensions
                                         (items, input_dim)
        y: List[List[Any]] = all expected output data of
                             dimensions (items, expected_dim)
    """
    def __init__(self) -> None:
        """
        Custom Initalizer for Dataset
        """
        self.x: List[List[Optional[float]]] = []
        self.y: List[List[Any]] = []
    
    def get_out(self) -> Tuple[Any]:
        """
        Public Output Value Getting Function:
            gets all unique y values
        
        Returns (name: type = use):
            _: tuple[Any] = all unique y values
        """
        out_arr: np.ndarray = np.asarray(self.y).flatten()
        return tuple(set(out_arr.tolist()))

class BinaryAddition(Dataset):
    """
    Binary Addition Data:
        all possible problems with addends 0 and 1
        input encoded as two one_hot sequences
        output encoded as sum as int
    """
    def __init__(self) -> None:
        """
        Custom Initalizer for Binary Addition
        """
        self.x: List[List[Optional[float]]] = [
            [1.0, None, 1.0, None],
            [1.0, None, None, 1.0],
            [None, 1.0, 1.0, None],
            [None, 1.0, None, 1.0]
        ]
        self.y: List[List[Any]] = [
            [0],
            [1],
            [1],
            [2]
        ]

class AnalogAddition(Dataset):
    """
    Analog Addition Data:
        all possible problems with addends 0 and 1
        input encoded as two floats
        output encoded as sum as int
    """
    def __init__(self) -> None:
        """
        Custom Initalizer for Analog Addition
        """
        self.x: List[List[Optional[float]]] = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ]
        self.y: List[List[Any]] = [
            [0],
            [1],
            [1],
            [2]
        ]

class ExtensiveAddition(Dataset):
    """
    Extensive Addition Data:
        all possible problems with addends 0 through 5
        input encoded as two one_hot sequences
        output encoded as sum as int
    """
    def __init__(self) -> None:
        """
        Custom Initalizer for Extensive Addition
        """
        _x: List[Tuple[int]] = [(int(combo[0]), int(combo[1])) for combo in combos('012345', 2)]
        self.x: List[List[Optional[float]]] = one_hot_int(_x)
        self.y: List[List[Any]] = [[int(sum(combo))] for combo in _x]

class ExtensiveAnalogAddition(Dataset):
    """
    Extensive Analog Addition Data:
       all possible problems with addends 0 through 5
       input encoded as two floats
       output encoded as sum as int
    """
    def __init__(self) -> None:
        """
        Custom Initializer for Extensive Addition
        """
        self.x: List[List[Optional[float]]] = [[float(combo[0]), float(combo[1])] for combo in combos('012345', 2)]
        self.y: List[List[Any]] = [[int(sum(combo))] for combo in self.x]

class EvenOddAnalog(Dataset):
    """
    Even Odd Data:
        numbers from 1 through 9
        input encoded as one float
        output bool representation of whether or not
        number is even
    """
    def __init__(self) -> None:
        """
        Custom Initializer for Even Odd
        """
        self.x: List[List[Optional[float]]] = [[float(num)] for num in range(1, 10)]
        self.y: List[List[Any]] = [[num[0] % 2 == 0] for num in self.x]

class ExtensiveEvenOddAnalog(Dataset):
    """
    Even Odd Data:
        numbers from 1 through 99
        input encoded as one float
        output bool representation of whether or not
        number is even
    """
    def __init__(self) -> None:
        """
        Custom Initializer for Even Odd
        """
        self.x: List[List[Optional[float]]] = [[float(num)] for num in range(1, 100)]
        self.y: List[List[Any]] = [[num[0] % 2 == 0] for num in self.x]