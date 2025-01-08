import numpy as np
from .gate import Gate

class Identity(Gate):
    def __init__(self, qubits=None):
        """
        Initialize the Identity (I) gate.

        :param qubits: Optional, indices of qubits this gate acts on.
        """
        super().__init__(name='I', qubits=qubits)
        self.matrix = np.array([
            [1, 0],
            [0, 1]
        ], dtype=complex)

    def apply(self, state_vector):
        """
        Apply the Identity gate to the given quantum state vector.

        :param state_vector: The state vector of the quantum system.
        :return: The unmodified state vector.
        """
        # The Identity gate does not modify the state vector
        return state_vector
