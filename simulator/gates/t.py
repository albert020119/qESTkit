import numpy as np
from .gate import Gate

class T(Gate):
    def __init__(self, qubits=None):
        """
        Initialize the T gate (π/8 phase gate).

        :param qubits: Optional, indices of qubits this gate acts on.
        """
        super().__init__(name='T', qubits=qubits)
        self.matrix = np.array([
            [1, 0],
            [0, np.exp(1j * np.pi / 4)]
        ], dtype=complex)

