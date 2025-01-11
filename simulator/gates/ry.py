import numpy as np
from .gate import Gate

class Ry(Gate):
    def __init__(self, theta, qubits=None):
        """
        Initialize the Ry gate (rotation around Y-axis).

        :param theta: Rotation angle in radians.
        :param qubits: Optional, indices of qubits this gate acts on.
        """
        super().__init__(name='Ry', qubits=qubits)
        self.theta = theta
        self.matrix = np.array([
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)]
        ], dtype=complex)
