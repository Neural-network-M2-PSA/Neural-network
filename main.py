import numpy as np
from typing import (Dict, Tuple, Callable,
                    Sequence, Iterator, NamedTuple)
from numpy import ndarray as Tensor

Func = Callable[[Tensor], Tensor]


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def test_alizee():
    predicted = Tensor
    actual = Tensor
    mse = MeanSquareError()
    predicted = [1, 1, 1, 1, 1, 1]
    actual = [1, 1, 2, 1, 3, 4]
    mse.loss(predicted, actual)
    print('loss:', mse)
    return mse


# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
    #print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#def test_Leila() :
    #print('coucou')
