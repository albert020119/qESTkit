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

    def apply(self, state_vector):
        """
        Apply the S gate to the given quantum state vector.

        :param state_vector: The state vector of the quantum system.
        :return: The modified state vector after applying the S gate.
        """
        if self.qubits is None or len(self.qubits) != 1:
            raise ValueError("S gate requires exactly one target qubit.")

        target_qubit = self.qubits[0]
        num_qubits = int(np.log2(len(state_vector)))

        if target_qubit >= num_qubits:
            raise ValueError("Target qubit index exceeds the number of qubits in the state vector.")

        # Construct the full operator by combining identity and S gate
        identity = np.eye(2, dtype=complex)
        operator = self.matrix
        for i in range(num_qubits):
            if i == target_qubit:
                continue
            operator = np.kron(identity, operator)

        return operator @ state_vector
