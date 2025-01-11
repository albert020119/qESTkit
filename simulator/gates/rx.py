import numpy as np
from .gate import Gate

class Rx(Gate):
    def __init__(self, theta, qubits=None):
        """
        Initialize the Rx gate (rotation around X-axis).

        :param theta: Rotation angle in radians.
        :param qubits: Optional, indices of qubits this gate acts on.
        """
        super().__init__(name='Rx', qubits=qubits)
        self.theta = theta
        self.matrix = np.array([
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)]
        ], dtype=complex)
