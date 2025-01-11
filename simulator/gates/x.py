import numpy as np

from .gate import Gate

class X(Gate):
    def __init__(self, qubits=None):
        """
        Initialize the Pauli-X gate.

        :param qubits: Optional, indices of qubits this gate acts on.
        """
        super().__init__(name='X', qubits=qubits)
        self.matrix = np.array([[0, 1],
                                [1, 0]], dtype=complex)

