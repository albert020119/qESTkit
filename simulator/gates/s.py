import numpy as np
from .gate import Gate

class S(Gate):
    def __init__(self, qubits=None):
        """
        Initialize the S gate (a special phase gate).

        :param qubits: Optional, indices of qubits this gate acts on.
        """
        super().__init__(name='S', qubits=qubits)
        self.matrix = np.array([
            [1, 0],
            [0, 1j]
        ], dtype=complex)

