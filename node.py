from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

from enums import Role

normcdf = norm().cdf

class Node(object):
    """
    ANN Node
    """
    def __init__(self, role: Optional[Role]=None, num: str="0", max_con: int=-1) -> None:
        """
        Custom Initializer for Node Object
        
        Parameters (name: type = use):
            role: Optional[Role] = type of node
            num: str = number or binding for naming
            max_con: int = maximum number of connections
        """
        self._num: str = num
        self._role: Role = role
        self._siblings: Dict[Node, float] = {}
        self._val: Optional[float] = None
        self._integrity: Dict[Node, float] = {}
        self._record: List[Tuple[Callable[[Iterable[Node], float], None]]] = []
        self._memory: Dict[Node, float] = {}
        self._spontaneity: float = -5
        self._max_con: int = max_con
        if self._role is Role.GROUND:
            no: Callable[..., None] = lambda *args, **kwargs: None
            self.step: Callable[..., bool] = lambda *args, **kwargs: True
            self.receive: Callable[..., None] = no
            self.connect: Callable[..., None] = no
    
    def set_max_con(self, max_con: int) -> None:
        """
        Maximum Connection Setter Function
        
        Parameters (name: type = use):
            max_con: int = maximum connections
        """
        self._max_con = max_con
    
    def get_connected_nodes(self) -> Iterable[Node]:
        """
        Connected Node Getter Function

        Returns (name: type = use):
            _: Iterable[Node] = all connected nodes
        """
        return self._siblings.keys()
    
    def role_str(self) -> str:
        """
        Custom Role String Function

        Returns (name: type = use):
            _: str = role
        """
        return str(self._role)[5:]
    
    def get_name(self) -> str:
        """
        Custom Name Function

        Returns (name: type = use):
            _: str = unique name
        """
        return self.role_str() + self._num
    
    def receive(self, val: Optional[float]) -> None:
        """
        Public Reception Function:
            called by other nodes to cause value transfer
        
        Paramters (name: type = use):
            val: float = the value to receive
        """
        self._val: Optional[float] = val if self._val is None or val is None else self._val + val
    
    def step(self, eval: bool=False, no_use: bool=False) -> bool:
        """
        Public Step Function:
            called by network every step
        
        Parameters (name: type = use):
            eval: bool = whether to run as eval without record
            no_use: bool = whether to disincentivize lack of use
        
        Returns:
            found: bool = whether or not a viable sibling was triggered
        """
        if self._val is not None:
            triggered: List[Node] = []
            found: bool = False
            for sibling in filter(lambda node: self._siblings[node] == self._val, ## KILL LESS THAN RULE
                                  self._siblings.keys()):
                found: bool = True
                triggered.append(sibling)
                if not eval:
                    self._record.append((partial(self._decrement, (sibling,)), partial(self._increment, (sibling,))))
                self.send(sibling, self._val)
            if not eval:
                not_triggered = [n for n in filter(lambda node: node not in triggered, self._siblings.keys())]
                if no_use:
                    self._decrement(not_triggered)
        else:
            found: bool = True
        if not eval:
            self._spontaneity += abs(np.random.normal()/100)
            if len(not_triggered) == 1 and self in not_triggered:
                self._spontaneity += 5
            p: float = normcdf(self._spontaneity)
            found = found if np.random.choice((True, False), p=(1 - p, p)) else False
        if found:
            self._val: Optional[float] = None
        elif (len(self._siblings) >= self._max_con) if self._max_con != -1 else False:
            self._val: Optional[float] = None
            found = True
        return found
    
    def reset_record(self) -> None:
        """
        Public Record Reset Function:
            resets the record of all triggered siblings
        """
        self._record = []
    
    def _decrement(self, nodes: Iterable[Node], base: float=10.0) -> None:
        """
        Private Integrity Decrementing Function
        
        Paramters (name: type = use):
            nodes: Iterable[Node] = list of nodes to be decremented
            base: float = number by which change is divided
        """
        actions = []
        for node in nodes:
            try:
                integrity: float = self._integrity[node] - abs(np.random.normal()/base)
                self._integrity[node] = integrity
                p: float = normcdf(integrity)
                actions.append(np.random.choice((partial(self._dissolve, node), partial(self._pass_over, node)), p=(1 - p, p)))
            except Exception as err:
                pass#print(err, end="")
        for action in actions:
            action()
        self._spontaneity += abs(np.random.normal()/1000)
    
    def _increment(self, nodes: Iterable[Node], base: float=10.0) -> None:
        """
        Private Integrity Incrementing Function
        
        Paramters (name: type = use):
            nodes: Iterable[Node] = list of nodes to be incremented
            base: float = number by which change is divided
        """
        for node in nodes:
            try:
                integrity: float = self._integrity[node] + abs(np.random.normal()/base)
                self._integrity[node] = integrity
            except Exception as err:
                pass#print(type(err), err)
        self._spontaneity -= abs(np.random.normal()/100)
    
    def _dissolve(self, node: Node) -> None:
        """
        Private Connection Dissolution Function:
            removes references to node in integrity and sibling
            dictionaries
        
        Paramters (name: type = use):
            node: Node = node with which connection is to be terminated
        """
        try:
            self._siblings.pop(node)
            self._memory[node] = self._integrity.pop(node)
        except:
            pass
    
    def _pass_over(self, node: Node) -> None:
        """
        Private Passing Over Function:
            counterpart to dissolve, placeholder for potential
            manipulations when a connection is not terminated
        
        Paramters (name: type = use):
            node: Node = node that was passed over
        """
        pass
    
    def send(self, receiver: Node, val: float) -> None:
        """
        Public Send Function:
            send a value to a receiving node
        
        Paramters (name: type = use):
            reciever: Node = receiving node
            val: float = value to be sent
        """
        receiver.receive(val)
    
    def connect(self, node: Node) -> None:
        """
        Public Connect Function:
            initalizes connection with another node
        
        Paramters (name: type = use):
            node: Node = node with which connection is to be initialized
        """
        #print(self._val, end=" ", flush=True)
        #for line in traceback.format_stack():
            #print(line.strip())
        self._siblings[node] = self._val
        if node not in self._memory:
            self._integrity[node] = np.random.normal() * 5
        else:
            self._integrity[node] = self._memory.pop(node)
        self._spontaneity -= 6
        self.send(node, self._val)
        self._val = None
    
    def reward(self, val: bool, base: float=10) -> None: ## add gradient calc
        """
        Public Rewarding Function:
            calls corresponding reward function for each recorded
            interaction
        
        Paramters (name: type = use):
            val: bool = whether or not an interaction is rewarded
            base: float = number by which change is divided
        """
        val: int = 1 if val else 0
        for record in self._record:
            record[val](base=base)
        self.reset_record()
    
    def is_active(self) -> bool:
        """
        Public Activation Monitor:
            return whether this node has received an unhadled val
        
        Returns (name: type = use):
            _: bool = whether or not this node is active and an
                      output node
        """
        return self._val is not None
    
    def get_role(self) -> Role:
        """
        Public Role Getting Function
        
        Returns (name: type = use):
            _: Role = node role
        """
        return self._role