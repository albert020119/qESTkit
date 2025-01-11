import numpy as np
from .gate import Gate

class Ph(Gate):
    def __init__(self, delta, qubits=None):
        """
        Initialize the Phase Shift (Ph) gate.

        :param delta: Phase shift angle in radians.
        :param qubits: Optional, indices of qubits this gate acts on.
        """
        super().__init__(name='Ph', qubits=qubits)
        self.delta = delta
        self.matrix = np.array([
            [1, 0],
            [0, np.exp(1j * delta)]
        ], dtype=complex)
