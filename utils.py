from typing import List, Optional, Tuple
from copy import copy

def one_hot_int(int_x: List[Tuple[int]]) -> List[List[Optional[int]]]:
    flat_x: List[int] = [item0[0] for item0 in int_x] + [item1[1] for item1 in int_x]
    half_x_size: int = max(flat_x) + 1
    half_x_template: List[None] = [None for _ in range(half_x_size)]
    out_x: List[List[Optional[int]]] = []
    for item in int_x:
        x_item0: List[Optional[int]] = copy(half_x_template)
        x_item1: List[Optional[int]] = copy(half_x_template)
        x_item0[item[0]] = 1
        x_item1[item[1]] = 1
        out_x.append(x_item0 + x_item1)
    return out_x