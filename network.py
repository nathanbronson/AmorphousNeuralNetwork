from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Tuple
from random import shuffle, randint
from pickle import load, dump

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import networkx as nx
from tqdm import tqdm

from enums import *
from node import Node

file_id = str(randint(0, 1e10))

output_favor = [
    ConnectionStrategy.OUTPUT_HALF,
    ConnectionStrategy.OUTPUT_75,
    ConnectionStrategy.OUTPUT_90,
    ConnectionStrategy.OUTPUT_99
]

class Network(object):
    """
    Amorphous Neural Network
    """
    def __init__(self, num_nodes: int, input_nodes: int, output_bindings: Iterable[Any], ground: bool=False) -> None:
        """
        Custom Initializer for Network Object
        
        Parameters (name: type = use):
            num_nodes: int = number of nodes making up the network
            input_nodes: int = number of input nodes
            output_bindings: Iterable[Any] = list of return values to be
                                             returned from step function
                                             if output node is activated
            ground: bool = whether or not to have a ground node
        """
        self._nodes: List[Node] = [Node(role=Role.NETWORK, num=str(num)) for num in range(num_nodes)]
        self._input_nodes: List[Node] = [Node(role=Role.INPUT, num=str(num)) for num in range(input_nodes)]
        self._output_nodes: Dict[Node, Any] = {Node(role=Role.OUTPUT, num=str(binding)): binding for binding in output_bindings}
        self._ground_node: Node = Node(role=Role.GROUND) if ground else None
        self._extra_nodes: List[Node] = []
        self._num_extra: int = 0
        self._steps: int = 0
        self._training_steps: int = 0
        self._connection_strategy: ConnectionStrategy = ConnectionStrategy.EVEN
        self._node_strategy: NodeStrategy = NodeStrategy.FIXED
        self._extra_handling: ExtraNodeHandling = ExtraNodeHandling.RESET
        self._reps: int = 0
        self._best: float = 0.0
        max_con = len(self._all_connectable_nodes()) - 1
        for node in self._all_nodes():
            node.set_max_con(max_con)
    
    def set_extra_node_handling(self, strat: ExtraNodeHandling) -> None:
        """
        Extra Node Handling Setting Function

        Parameters (name: type = use):
            strat: ExtraNodeHandling = strategy to be used
        """
        self._extra_handling: ExtraNodeHandling = strat
    
    def set_node_strategy(self, strat: NodeStrategy) -> None:
        """
        Node Strategy Setting Function

        Parameters (name: type = use):
            strat: NodeStrategy = strategy to be used
        """
        self._node_strategy: NodeStrategy = strat

    def set_connection_strategy(self, strat: ConnectionStrategy) -> None:
        """
        Connection Strategy Setting Function
        
        Parameters (name: type = use):
            strat: ConnectionStrategy = strategy to be used
        """
        self._connection_strategy: ConnectionStrategy = strat
    
    def reset_state(self) -> None:
        """
        Reset State Function:
            sets each node value to None
        """
        for node in self._all_nodes():
            node.receive(None)
    
    def _step(self, input: Iterable[float]=None, eval: bool=False) -> List[Any]:
        """
        Private Step Function for the Network:
            calls public step function of each active node in reverse
        
        Parameters (name: type = use):
            input: Iterable[float] = input values to be given to input
                                     nodes
            eval: bool = whether to run step function as eval without
                         record
        
        Returns:
            ret: List[Any] = all output bindings of active output nodes
        """
        self._steps += 1
        ret: List[Any] = []
        if input is not None:
            for (val, input_node) in zip(input, self._input_nodes):
                input_node.receive(val)
        for node in filter(lambda node: node.is_active(),
                           self._input_nodes + self._nodes[::-1]):
            if not node.step(eval=eval) and not eval:
                node.connect(self._random_connectable_node(exclude=node.get_connected_nodes()))
        for output_node in filter(lambda out_node: out_node.is_active(),
                                  self._output_nodes.keys()):
            ret.append(self._output_nodes[output_node])
            output_node.receive(None)
        return ret
    
    def _step_to(self, input: Iterable[float], eval: bool=False, max_step: int=1e4, clean: bool=True) -> List[Any]:
        """
        Private Continuous Step Function:
            calls private step function until a non-empty return is
            received
        
        Parameters (name: type = use):
            input: Iterable[float] = input values to be passed to private
                                     step function
            eval: bool = whether to run step function as eval without
                         record
            max_step: int = maximum number of steps until return [None]
            clean: bool = whether state should be reset after each step
        
        Returns:
            ret: List[Any] = output received from private step function
        """
        self.reset_state()
        ret: List[Any] = []
        ret: List[Any] = self._step(input=input, eval=eval)
        _step: int = 0
        while not ret:
            _step += 1
            ret: List[Any] = self._step(eval=eval)
            if not ret and _step >= max_step:
                ret: List[Any] = [None]
        if clean:
            self.reset_state()
        return ret
    
    def _step_for(self) -> None:
        """
        """
        raise NotImplementedError
    
    def _reward(self, val: bool, base: float=10) -> None:
        """
        Apply Binary Reward (True positive, False negative)
        
        Paramters (name: type = use):
            val: bool = binary reward
            base: float = number by which change is divided
        """
        for node in self._all_nodes():
            node.reward(val, base=base)
    
    def _bp_train_step(self, input: Iterable[float], expected: List[Any], clean: bool=True, can_increment: bool=True) -> None:
        """
        Private Backpropogate Training Step Function:
            applies model to input then rewards based on output similarity
            to expected
        
        Parameters (name: type = use):
            input: Iterable[float] = list of input values to be passed to
                                     the model
            expected: List[Any] = expected output values of the model
            clean: bool = whether state should be reset after each step
            can_increment: bool = whether or not a connection can be
                                  incremented
        """
        self._training_steps += 1
        actual = self._step_to(input, clean=clean)
        if set(expected) == set(actual) and can_increment:
            self._reward(True)
        elif set(expected) != set(actual):
            self._reward(False)
    
    def _all_connectable_nodes(self) -> List[Node]:
        """
        Connectable Node Getting Function:
            returns network, output, and ground nodes
        
        Returns (name: type = use):
            _: List[Node] = all connectable nodes
        """
        return self._nodes + list(self._output_nodes.keys()) + ([self._ground_node] if self._ground_node is not None else []) + self._extra_nodes
    
    def _all_nodes(self) -> List[Node]:
        """
        Node Getting Function:
            returns all nodes in network
        
        Returns (name: type = use):
            _: List[Node] = all nodes
        """
        return self._all_connectable_nodes() + self._input_nodes

    def _random_connectable_node(self, exclude: Iterable[Node]=(), use_strategy: bool=True) -> Node:
        """
        Random Connectable Node Retrieval

        Parameters (name: type = use):
            exclude: Iterable[Node] = nodes to exclude
            use_strategy: bool = whether or not the connection
                                 strategy should be used
        """
        if self._connection_strategy is ConnectionStrategy.EVEN or not use_strategy:
            return np.random.choice([node for node in filter(lambda e: e not in list(exclude), self._all_connectable_nodes())])
        elif self._connection_strategy in output_favor:
            val: float = [.5, .75, .90, .99][output_favor.index(self._connection_strategy)]
            nodes: List[Node] = [node for node in filter(lambda e: e not in list(exclude), self._all_connectable_nodes())]
            role: List[Role] = [item for item in map(lambda node: node.get_role(), nodes)]
            if role.count(Role.OUTPUT) > 0 and role.count(Role.NETWORK) > 0:
                out_val: float = (val)/role.count(Role.OUTPUT)
                net_val: float = (1 - val)/role.count(Role.NETWORK)
            elif role.count(Role.OUTPUT) > 0:
                out_val: float = 1/role.count(Role.OUTPUT)
                net_val: float = 0.0
            elif role.count(Role.NETWORK) > 0:
                out_val: float = 0.0
                net_val: float = 1/role.count(Role.NETWORK)
            p: List[float] = [item for item in map(lambda node: out_val if node.get_role() is Role.OUTPUT else net_val, nodes)]
            return np.random.choice(nodes, p=p)
    
    def fit(self, technique: Technique, x: Iterable[float], y: Iterable[Any], *args: Tuple[Any], **kwargs: Dict[str, Any]) -> None:
        """
        Main Fit Function:
            fits model with given data, technique, and args/kwargs
        
        Parameter (name: type = use):
            x: Iterable[float] = input data of dimensions
                                 (items, input_dim) for fitting
            y: Iterable[Any] = expected output data of dimensions
                               (items, expected_dim) for fitting
            *args: Tuple[Any] = all extra arguments for specific to
                                technique
            **kwargs: Dict[str, Any] = all extra keyword arguments
                                       specific to technique
        """
        techniques: Dict[Technique, Callable[..., None]] = {
            Technique.BACKPROPOGATE: self._bp_fit,
            Technique.GAIN_OF_FUNCTION: self._gof_fit,
            Technique.BPGOF: self._bpgof_fit
        }
        techniques[technique](x, y, *args, **kwargs)
    
    def _bpgof_fit(self, x: Iterable[float], y: Iterable[Any], metrics: Iterable[Metric]=(), end_at: int=-1, log_interval: int=1e4) -> None:
        """
        Main BPGOF Fit Function:
            fits function to data using blend of gain of function and
            backpropgate strategies
        
        Parameters (name: type = use):
            x: Iterable[float] = input data of dimensions
                                 (items, input_dim) for fitting
            y: Iterable[Any] = expected output data of dimensions
                               (items, expected_dim) for fitting
            metrics: Iterable[Metric] = all metrics to be printed at
                                        specified log interval
            end_at: int = number of total network training steps to
                          end at (not implemented)
            log_interval: int = interval at which metric data will be
                                printed
        """
        while True:
            self._gof_fit(x, y, metrics=metrics, end_at=self._training_steps + 5e4, need_all=False)
            self._bp_fit(x, y, metrics=metrics, end_at=self._training_steps + 1e4, log_interval=log_interval, can_increment=False)
    
    def _gof_train_step(self, input: Iterable[float], expected: Iterable[Any], clean: bool=True) -> bool:
        """
        Private Gain of Function Training Step Function:
            applies model to input then returns similarity
        
        Parameters (name: type = use):
            input: Iterable[float] = list of input values to be passed to
                                     the model
            expected: List[Any] = expected output values of the model
            clean: bool = whether state should be reset after each step
        
        Returns:
            _: bool = whether the model was accurate
        """
        self._training_steps += 1
        return set(expected) == set(self._step_to(input, clean=True))
    
    def _gof_eval(self, input: Iterable[float], expected: Iterable[Any], clean: bool=True) -> bool:
        """
        Private Gain of Function Evaluation Function:
            determines whether actual is expected without training
        
        Parameters (name: type = use):
            input: Iterable[float] = list of input values to be passed to
                                     the model
            expected: List[Any] = expected output values of the model
            clean: bool = whether state should be reset after each step
        
        Returns:
            _: bool = whether the model was accurate
        """
        return set(expected) == set(self._step_to(input, clean=clean, eval=True))
    
    def reset_record(self) -> None:
        """
        """
        for node in self._all_nodes():
            node.reset_record()

    def _gof_decay(self, x: Iterable[float], y: Iterable[any]) -> None:
        """
        """
        self.reset_record()
        for _x, _y in zip(x, y):
            if set(self._step_to(_x, clean=True, eval=True)) != set(_y):
                self._reward(False, base=0.75)
    
    def _gof_fit(self, x: Iterable[float], y: Iterable[Any], metrics: Iterable[Metric]=(), attack: int=-1, end_at: int=-1, resume: bool=False, need_all: bool=True, plateau_thresh: int=-1, attack_strat: AttackStrategy=AttackStrategy.RESTART, log_interval: int=1e3, save_at_best: bool=False) -> None:
        """
        Main Gain of Function Fit Function:
            fits function to data using gain of fit technique
        
        Parameters (name: type = use):
            x: Iterable[float] = input data of dimensions
                                 (items, input_dim) for fitting
            y: Iterable[Any] = expected output data of dimensions
                               (items, expected_dim) for fitting
            metrics: Iterable[Metric] = all metrics to be printed at
                                        specified log interval
            attack: int = threshold at which the attack strategy
                           (repeatedly restarting fit) should be used
            end_at: int = number of total network training steps to
                          end at
            resume: int = whether or not to resume from saved model
            plateau_thresh: int = threshold at which a plateau is said
                                  to have begun
            attack_strat: AttackStrategy = action to be taken when
                                           attack threshold is reached
            log_interval: int = interval at which metric data will be
                                printed
            save_at_best: bool = whether or not to save at best result
                                 regardless of where improvement is 
                                 made
        """
        self.x, self.y = x, y
        data: List[Tuple[float, Any]] = [point for point in zip(x, y)]
        count: int = 0
        attacked: bool = False
        if resume:
            self.transfer_data(load_model())
        while True:
            shuffle(data)
            done: List[List[Any]] = [[], []]
            self._commit()
            for (n, point) in enumerate(data):
                count: int = 0
                while True:
                    if self._training_steps % log_interval == 0:
                        print(*(["point:", str(n)] + self.eval_metrics(x, y, metrics)), flush=True)
                    if end_at != -1 and self._training_steps >= end_at:
                        return
                    count += 1
                    if save_at_best:
                        cur: float = float(self.eval_metrics(x, y, (Metric.ACCURACY,))[-1])
                        if cur > self._best:
                            self._commit()
                            self._best: float = cur
                    if self._gof_train_step(point[0], point[1]) \
                    and ((all(map(self._gof_eval, *done)) if need_all else True)):
                        self._commit()
                        break
                    self._revert()
                    attacked: bool = count >= attack and attack != -1
                    if attacked:
                        if attack_strat is AttackStrategy.RESTART:
                            break
                        elif attack_strat is AttackStrategy.DECAY:
                            self._gof_decay(x, y)
                    plateaued: bool = count >= plateau_thresh and plateau_thresh != -1
                    if plateaued:
                        if self._node_strategy is NodeStrategy.ADD_PLATEAU:
                            self._num_extra += 1
                            self._extra_nodes.append(Node(role=Role.NETWORK, num=str(len(self._nodes))))
                if attacked and attack_strat is AttackStrategy.RESTART:
                    break
                done[0].append(point[0])
                done[1].append(point[1])
            if attacked and attack_strat is AttackStrategy.RESTART:
                break
            print(*self.eval_metrics(x, y, metrics), flush=True)
            self._reps += 1
        if attacked and attack_strat is AttackStrategy.RESTART:
            self.transfer_data(Network(len(self._nodes), len(self._input_nodes), list(self._output_nodes.values())))
        self._gof_fit(x, y, metrics=metrics, attack=attack, need_all=need_all, plateau_thresh=plateau_thresh, attack_strat=attack_strat)
    
    def _commit(self) -> None:
        """
        Private Commit Function:
            saves network data for gain of function training
        """
        #print(*(["commit:"]+self.eval_metrics(self.x, self.y, (Metric.ACCURACY,))), flush=True)
        self.save()
    
    def _revert(self) -> None:
        """
        Private Revert Function:
            reverts to network saved data
        """
        self.transfer_data(load_model())
        if self._extra_handling is ExtraNodeHandling.SAVE:
            while len(self._extra_nodes) < self._num_extra:
                self._extra_nodes.append(Node(role=Role.NETWORK, num=str(len(self._extra_nodes) + len(self._nodes))))
    
    def transfer_data(self, source: Network) -> None:
        """
        Transfer Function:
            moves data from a network to self
        
        Parameter (name: type = use):
            source: Network = source network from which data should be
                              transferred
        """
        self._ground_node = source._ground_node
        self._input_nodes = source._input_nodes
        self._nodes = source._nodes
        self._output_nodes = source._output_nodes
        self._extra_nodes = source._extra_nodes
    
    def _bp_fit(self, x: Iterable[float], y: Iterable[Any], metrics: Iterable[Metric]=(), log_interval: int=100, end_at: int=-1, can_increment: bool=True) -> None:
        """
        Main Backpropogate Fit Function:
            fits model to data x and y using backpropogation
        
        Parameter (name: type = use):
            x: Iterable[float] = input data of dimensions
                                 (items, input_dim) for fitting
            y: Iterable[Any] = expected output data of dimensions
                               (items, expected_dim) for fitting
            metrics: Iterable[Metric] = all metrics to be printed at
                                        specified log interval
            log_interval: int = interval at which metric data will be
                                printed
            end_at: int = number of total network training steps to
                          end at
            can_increment: bool = whether or not a connection can be
                                  incremented
        """
        data: List[Tuple[float, Any]] = [point for point in zip(x, y)]
        train_step: int = 0
        while True:
            if end_at != -1 and self._training_steps >= end_at:
                return
            train_step += 1
            idx: int = np.random.choice(range(len(x)))
            self._bp_train_step(data[idx][0], data[idx][1], can_increment=can_increment)
            if train_step % log_interval == 0:
                print(*self.eval_metrics(x, y, metrics), flush=True)
    
    def eval_metrics(self, x: Iterable[float], y: Iterable[Any], metrics: Iterable[Metric]) -> List[float]:
        """
        Metric Evaluation Function
        
        Parameters (name: type = use):
            x: Iterable[float] = input data of dimensions
                                 (items, input_dim) for evaluation
            y: Iterable[Any] = expected output data of dimensions
                               (items, expected_dim) for evaluation
            metrics: Iterable[Metric] = metric to be evaluated
        
        Returns:
            evals: List[str] = a list of all metrics that were evaluated
        """
        evals: List[str] = []
        for metric in metrics:
            if metric is Metric.ACCURACY:
                count: List[int] = [0, 0]
                for pair in zip(x, y):
                    if set(pair[1]) == set(self._step_to(pair[0], eval=True, clean=True)):
                        count[0] += 1
                    else:
                        count[1] += 1
                total: int = sum(count)
                evals.append("accuracy:")
                evals.append(f"{count[0]/total:.3f}")
            elif metric is Metric.STEP:
                evals.append("model step:")
                evals.append(str(self._steps))
            elif metric is Metric.TRAINING_STEP:
                evals.append("training step:")
                evals.append(str(self._training_steps))
            elif metric is Metric.NODES:
                evals.append("nodes:")
                evals.append(str(len(self._all_nodes())))
            elif metric is Metric.REPS:
                evals.append("reps:")
                evals.append(str(self._reps))
        return evals
    
    def run(self) -> None:
        """
        Main Run Function for Network:
            repeatedly calls private step function
        """
        while True:
            self._step()
    
    def visualize(self) -> None:
        """
        Visualize Function:
            shows node plot representation of network
        """
        graph: Dict[str, List[str]] = {en[1].get_name(): [(node.get_name(), en[1]._integrity[node], en[1]._siblings[node]) for node in list(en[1]._integrity.keys())] for en in enumerate(self._all_nodes())}
        _from: List[str] = []
        _to: List[str] = []
        for f in list(graph.keys()):
            for t in graph[f]:
                _from.append(f)
                _to.append(t)
        df: pd.DataFrame = pd.DataFrame({"to": [item[0] for item in _to], "from": _from})
        graph = nx.from_pandas_edgelist(df, "from", "to", create_using=nx.DiGraph().reverse())
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos=pos, with_labels=True, node_size=500, node_color="skyblue", node_shape="s", alpha=0.75, linewidths=40, arrows=True)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels={(item[0], item[1][0]): f"{item[1][1]:.2f}, {item[1][2]:.2f}" for item in zip(_from, _to)}, font_color='red')
        plt.show()
    
    def save(self, path: str="./model{}.ann".format(file_id)) -> None:
        with open(path, "wb+") as doc:
            dump(self, doc)

def load_model(path: str="./model{}.ann".format(file_id)) -> Network:
    with open(path, "rb") as model:
        return load(model)