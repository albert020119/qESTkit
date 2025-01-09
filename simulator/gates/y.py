import numpy as np

from simulator.gates.gate import Gate


class Y(Gate):
    def __init__(self, qubits):
        """
        Initialize the Y gate.

        :param qubits: List of qubit indices this gate acts on. Should contain exactly one index.
        """
        super().__init__(name="Y", qubits=qubits)
        if len(self.qubits) != 1:
            raise ValueError("Y gate acts on exactly one qubit.")
        # The matrix representation of the Y gate
        self.matrix = np.array([[0, -1j],
                                [1j, 0]], dtype=complex)

    def apply(self, state_vector):
        """
        Apply the Y gate to a quantum state vector.

        :param state_vector: The state vector of the quantum system.
        :return: The modified state vector after applying the gate.
        """
        if self.qubits is None or len(self.qubits) != 1:
            raise ValueError("Rz gate requires exactly one target qubit.")

        target_qubit = self.qubits[0]
        num_qubits = int(np.log2(len(state_vector)))

        if target_qubit >= num_qubits:
            raise ValueError("Target qubit index exceeds the number of qubits in the state vector.")

        # Construct the full operator by combining identity and Rz gate
        identity = np.eye(2, dtype=complex)
        operator = self.matrix
        for i in range(num_qubits):
            if i == target_qubit:
                continue
            operator = np.kron(identity, operator)

        return operator @ state_vector
