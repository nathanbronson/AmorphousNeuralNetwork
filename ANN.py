from __future__ import annotations

from datasets import *
from enums import *
from network import *
from node import *
from utils import *

def analog_addition_train():
    try:
        dataset: Dataset = AnalogAddition()
        add_net: Network = Network(10, 2, dataset.get_out())
        add_net.fit(Technique.BACKPROPOGATE, dataset.x, dataset.y, metrics=(Metric.ACCURACY, Metric.STEP, Metric.TRAINING_STEP), log_interval=1e2)
    except KeyboardInterrupt:
        try:
            print("", end="\r")
            print("0 + 0", add_net._step_to((0, 0), eval=True, clean=True))
            print("0 + 1", add_net._step_to((0, 1), eval=True, clean=True))
            print("1 + 0", add_net._step_to((1, 0), eval=True, clean=True))
            print("1 + 1", add_net._step_to((1, 1), eval=True, clean=True))
            add_net.visualize()
        except KeyboardInterrupt:
            add_net.visualize()

def digital_addition_train():
    try:
        dataset: Dataset = BinaryAddition()
        add_net: Network = Network(10, 4, dataset.get_out())
        add_net.fit(Technique.BACKPROPOGATE, dataset.x, dataset.y, metrics=(Metric.ACCURACY, Metric.STEP, Metric.TRAINING_STEP), log_interval=1e3)
    except KeyboardInterrupt:
        try:
            print("", end="\r")
            print("0 + 0", add_net._step_to([1, None, 1, None], eval=True, clean=True))
            print("0 + 1", add_net._step_to([1, None, None, 1], eval=True, clean=True))
            print("1 + 0", add_net._step_to([None, 1, 1, None], eval=True, clean=True))
            print("1 + 1", add_net._step_to([None, 1, None, 1], eval=True, clean=True))
            add_net.visualize()
        except KeyboardInterrupt:
            add_net.visualize()

def digital_gof_addition_train():
    try:
        dataset: Dataset = BinaryAddition()
        add_net: Network = Network(10, 4, dataset.get_out())
        add_net.fit(Technique.GAIN_OF_FUNCTION, dataset.x, dataset.y, metrics=(Metric.ACCURACY, Metric.STEP, Metric.TRAINING_STEP), attack=100)
    except KeyboardInterrupt:
        try:
            print("", end="\r")
            print("0 + 0", add_net._step_to([1, None, 1, None], eval=True, clean=True))
            print("0 + 1", add_net._step_to([1, None, None, 1], eval=True, clean=True))
            print("1 + 0", add_net._step_to([None, 1, 1, None], eval=True, clean=True))
            print("1 + 1", add_net._step_to([None, 1, None, 1], eval=True, clean=True))
            add_net.visualize()
        except KeyboardInterrupt:
            add_net.visualize()

def extensive_digital_gof_addition_train():
    try:
        dataset: Dataset = ExtensiveAddition()
        add_net: Network = Network(75, 4, dataset.get_out())
        add_net.set_connection_strategy(ConnectionStrategy.OUTPUT_HALF)
        add_net.fit(Technique.GAIN_OF_FUNCTION, dataset.x, dataset.y, metrics=(Metric.ACCURACY, Metric.STEP, Metric.TRAINING_STEP, Metric.NODES), attack=250)
    except KeyboardInterrupt:
        try:
            print("", end="\r")
            print("0 + 0", add_net._step_to([1, None, 1, None], eval=True, clean=True))
            print("0 + 1", add_net._step_to([1, None, None, 1], eval=True, clean=True))
            print("1 + 0", add_net._step_to([None, 1, 1, None], eval=True, clean=True))
            print("1 + 1", add_net._step_to([None, 1, None, 1], eval=True, clean=True))
            add_net.visualize()
        except KeyboardInterrupt:
            add_net.visualize()

def extensive_addition_train():
    try:
        dataset: Dataset = ExtensiveAddition()
        add_net: Network = Network(10, 4, dataset.get_out())
        add_net.fit(Technique.BACKPROPOGATE, dataset.x, dataset.y, metrics=(Metric.ACCURACY, Metric.STEP, Metric.TRAINING_STEP), log_interval=5e3)
    except KeyboardInterrupt:
        try:
            print("", end="\r")
            print("0 + 0", add_net._step_to([1, None, 1, None], eval=True, clean=True))
            print("0 + 1", add_net._step_to([1, None, None, 1], eval=True, clean=True))
            print("1 + 0", add_net._step_to([None, 1, 1, None], eval=True, clean=True))
            print("1 + 1", add_net._step_to([None, 1, None, 1], eval=True, clean=True))
            add_net.visualize()
        except KeyboardInterrupt:
            add_net.visualize()

def test_save_load():
    dataset: Dataset = BinaryAddition()
    add_net: Network = Network(100, 4, dataset.get_out())
    try:
        add_net.fit(Technique.BACKPROPOGATE, dataset.x, dataset.y, metrics=(Metric.ACCURACY, Metric.STEP, Metric.TRAINING_STEP), log_interval=1e3)
    except KeyboardInterrupt:
        print("", end="\r")
        print("=" * 20, "ADD_NET", "=" * 20)
        for _ in range(3):
            print("0 + 0", add_net._step_to([1, None, 1, None], eval=True))
            print("0 + 1", add_net._step_to([1, None, None, 1], eval=True))
            print("1 + 0", add_net._step_to([None, 1, 1, None], eval=True))
            print("1 + 1", add_net._step_to([None, 1, None, 1], eval=True))
        print("=" * 20, "ADD_NET", "=" * 20)
    add_net.save()
    load_net = load_model()
    print("=" * 20, "LOAD_NET", "=" * 20)
    for _ in range(3):
        print("0 + 0", load_net._step_to([1, None, 1, None], eval=True))
        print("0 + 1", load_net._step_to([1, None, None, 1], eval=True))
        print("1 + 0", load_net._step_to([None, 1, 1, None], eval=True))
        print("1 + 1", load_net._step_to([None, 1, None, 1], eval=True))
    print("=" * 20, "LOAD_NET", "=" * 20)

def extensive_digital_bpgof_addition_train():
    try:
        dataset: Dataset = ExtensiveAddition()
        add_net: Network = Network(75, 4, dataset.get_out())
        add_net.set_connection_strategy(ConnectionStrategy.OUTPUT_HALF)
        add_net.fit(Technique.BPGOF, dataset.x, dataset.y, metrics=(Metric.ACCURACY, Metric.STEP, Metric.TRAINING_STEP))
    except KeyboardInterrupt:
        try:
            print("", end="\r")
            print("0 + 0", add_net._step_to([1, None, 1, None], eval=True, clean=True))
            print("0 + 1", add_net._step_to([1, None, None, 1], eval=True, clean=True))
            print("1 + 0", add_net._step_to([None, 1, 1, None], eval=True, clean=True))
            print("1 + 1", add_net._step_to([None, 1, None, 1], eval=True, clean=True))
            add_net.visualize()
        except KeyboardInterrupt:
            add_net.visualize()

def analog_addition_gof_train():
    try:
        dataset: Dataset = AnalogAddition()
        add_net: Network = Network(10, 2, dataset.get_out())
        add_net.fit(Technique.GAIN_OF_FUNCTION, dataset.x, dataset.y, metrics=(Metric.ACCURACY, Metric.STEP, Metric.TRAINING_STEP), attack=1e3)
    except KeyboardInterrupt:
        try:
            print("", end="\r")
            print("0 + 0", add_net._step_to((0, 0), eval=True, clean=True))
            print("0 + 1", add_net._step_to((0, 1), eval=True, clean=True))
            print("1 + 0", add_net._step_to((1, 0), eval=True, clean=True))
            print("1 + 1", add_net._step_to((1, 1), eval=True, clean=True))
            add_net.visualize()
        except KeyboardInterrupt:
            add_net.visualize()

def extensive_analog_addition_gof_train():
    try:
        dataset: Dataset = ExtensiveAnalogAddition()
        add_net: Network = Network(50, 2, dataset.get_out())
        #add_net.set_connection_strategy(ConnectionStrategy.OUTPUT_HALF)
        #add_net.set_node_strategy(NodeStrategy.ADD_PLATEAU)
        add_net.fit(Technique.GAIN_OF_FUNCTION, dataset.x, dataset.y, metrics=(Metric.ACCURACY, Metric.STEP, Metric.TRAINING_STEP, Metric.REPS), attack=1e4, attack_strat=AttackStrategy.RESTART, log_interval=5e3)
    except KeyboardInterrupt:
        try:
            print("", end="\r")
            print("0 + 0", add_net._step_to((0, 0), eval=True, clean=True))
            print("0 + 1", add_net._step_to((0, 1), eval=True, clean=True))
            print("1 + 0", add_net._step_to((1, 0), eval=True, clean=True))
            print("1 + 1", add_net._step_to((1, 1), eval=True, clean=True))
            add_net.visualize()
        except KeyboardInterrupt:
            add_net.visualize()

def extensive_analog_addition_bpgof_train():
    try:
        dataset: Dataset = ExtensiveAnalogAddition()
        add_net: Network = Network(75, 4, dataset.get_out())
        add_net.set_connection_strategy(ConnectionStrategy.OUTPUT_HALF)
        add_net.fit(Technique.BPGOF, dataset.x, dataset.y, metrics=(Metric.ACCURACY, Metric.STEP, Metric.TRAINING_STEP))
    except KeyboardInterrupt:
        try:
            print("", end="\r")
            print("0 + 0", add_net._step_to((0, 0), eval=True, clean=True))
            print("0 + 1", add_net._step_to((0, 1), eval=True, clean=True))
            print("1 + 0", add_net._step_to((1, 0), eval=True, clean=True))
            print("1 + 1", add_net._step_to((1, 1), eval=True, clean=True))
            add_net.visualize()
        except KeyboardInterrupt:
            add_net.visualize()

def even_odd_analog_gof_train():
    try:
        dataset: Dataset = EvenOddAnalog()
        add_net: Network = Network(20, 1, dataset.get_out())
        add_net.set_connection_strategy(ConnectionStrategy.OUTPUT_99)
        add_net.fit(Technique.GAIN_OF_FUNCTION, dataset.x, dataset.y, metrics=(Metric.ACCURACY, Metric.STEP, Metric.TRAINING_STEP, Metric.REPS), log_interval=1e4)
    except KeyboardInterrupt:
        try:
            print("", end="\r")
            for i in range(1, 10):
                print(str(i), add_net._step_to((i,), eval=True, clean=True))
            add_net.visualize()
        except KeyboardInterrupt:
            add_net.visualize()

def extensive_even_odd_analog_gof_train():
    try:
        dataset: Dataset = ExtensiveEvenOddAnalog()
        add_net: Network = Network(150, 1, dataset.get_out())
        add_net.set_connection_strategy(ConnectionStrategy.OUTPUT_99)
        add_net.fit(Technique.GAIN_OF_FUNCTION, dataset.x, dataset.y, metrics=(Metric.ACCURACY, Metric.STEP, Metric.TRAINING_STEP, Metric.REPS), log_interval=1e3)
    except KeyboardInterrupt:
        try:
            print("", end="\r")
            for i in range(1, 10):
                print(str(i), add_net._step_to((i,), eval=True, clean=True))
            add_net.visualize()
        except KeyboardInterrupt:
            add_net.visualize()

if __name__ == "__main__":
    from sys import argv
    {
        "ext": extensive_addition_train,
        "digital": digital_addition_train,
        "digitalgof": digital_gof_addition_train,
        "extdigitalgof": extensive_digital_gof_addition_train,
        "extdigitalbpgof": extensive_digital_bpgof_addition_train,
        "analoggof": analog_addition_gof_train,
        "extanaloggof": extensive_analog_addition_gof_train,
        "extanalogbpgof": extensive_analog_addition_bpgof_train,
        "eoanaloggof": even_odd_analog_gof_train,
        "exteoanaloggof": extensive_even_odd_analog_gof_train
    }              \
    [str(argv[1])]()
