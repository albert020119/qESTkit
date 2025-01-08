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

    def apply(self, state_vector):
        """
        Apply the Hadamard gate to the given quantum state vector.

        :param state_vector: The state vector of the quantum system.
        :return: The modified state vector after applying the Hadamard gate.
        """
        if self.qubits is None or len(self.qubits) != 1:
            raise ValueError("Hadamard gate requires exactly one target qubit.")

        target_qubit = self.qubits[0]
        num_qubits = int(np.log2(len(state_vector)))

        if target_qubit >= num_qubits:
            raise ValueError("Target qubit index exceeds the number of qubits in the state vector.")

        # Construct the full operator by combining identity and Hadamard gate
        identity = np.eye(2, dtype=complex)
        operator = self.matrix
        for i in range(num_qubits):
            if i == target_qubit:
                continue
            operator = np.kron(identity, operator)

        return operator @ state_vector
