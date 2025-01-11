import numpy as np
from .gate import Gate

class Hadamard(Gate):
    def __init__(self, qubits=None):
        """
        Initialize the Hadamard gate.

        :param qubits: Optional, indices of qubits this gate acts on.
        """
        super().__init__(name='H', qubits=qubits)
        self.matrix = np.array([
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
            [1 / np.sqrt(2), -1 / np.sqrt(2)]
        ], dtype=complex)
        self.name = "Hadamard"