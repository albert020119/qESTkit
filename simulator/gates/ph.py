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

    def apply(self, state_vector):
        """
        Apply the Phase Shift (Ph) gate to the given quantum state vector.

        :param state_vector: The state vector of the quantum system.
        :return: The modified state vector after applying the Ph gate.
        """
        if self.qubits is None or len(self.qubits) != 1:
            raise ValueError("Ph gate requires exactly one target qubit.")

        target_qubit = self.qubits[0]
        num_qubits = int(np.log2(len(state_vector)))

        if target_qubit >= num_qubits:
            raise ValueError("Target qubit index exceeds the number of qubits in the state vector.")

        # Construct the full operator by combining identity and Ph gate
        identity = np.eye(2, dtype=complex)
        operator = self.matrix
        for i in range(num_qubits):
            if i == target_qubit:
                continue
            operator = np.kron(identity, operator)

        return operator @ state_vector
