import numpy as np
from .gate import Gate

class Rz(Gate):
    def __init__(self, theta, qubits=None):
        """
        Initialize the Rz gate (rotation around Z-axis).

        :param theta: Rotation angle in radians.
        :param qubits: Optional, indices of qubits this gate acts on.
        """
        super().__init__(name='Rz', qubits=qubits)
        self.theta = theta
        self.matrix = np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=complex)

