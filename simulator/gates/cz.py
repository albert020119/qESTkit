import numpy as np
from .gate import Gate

class CZ(Gate):
    def __init__(self, qubits=None):
        """
        Initialize the Controlled-Z gate.

        :param qubits: Indices of qubits this gate acts on. Should contain exactly two qubits: [control, target].
        """
        if qubits is None or len(qubits) != 2:
            raise ValueError("CZ gate requires exactly two qubits: [control, target].")
        super().__init__(name='CZ', qubits=qubits)

        # CZ gate matrix (acts on 2 qubits)
        self.matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)

    def apply(self, state_vector):
        """
        Apply the CZ gate to the given quantum state vector.

        :param state_vector: The state vector of the quantum system.
        :return: The modified state vector after applying the CZ gate.
        """
        if len(self.qubits) != 2:
            raise ValueError("CZ gate requires exactly two qubits.")

        control, target = self.qubits
        num_qubits = int(np.log2(len(state_vector)))

        # Validate qubit indices
        if control >= num_qubits or target >= num_qubits:
            raise ValueError("Control or target qubit index exceeds the number of qubits in the state vector.")

        if control == target:
            raise ValueError("Control and target qubits cannot be the same.")

        # Build the full operator for the CZ gate
        operator = self.get_operator(num_qubits)
        return operator @ state_vector

    def get_operator(self, num_qubits):
        """
        Construct the full operator for the CZ gate in a multi-qubit system.

        :param num_qubits: Total number of qubits in the system.
        :return: A matrix representing the CZ operator for the full system.
        """
        if len(self.qubits) != 2:
            raise ValueError("CZ gate requires exactly two qubits: control and target.")

        control, target = self.qubits

        if control >= num_qubits or target >= num_qubits:
            raise ValueError("Control or target qubit index exceeds the number of qubits in the state vector.")

        if control == target:
            raise ValueError("Control and target qubits cannot be the same.")

        # Total dimension of the state space
        dim = 2 ** num_qubits

        # Initialize the full CZ operator as an identity matrix
        full_operator = np.eye(dim, dtype=complex)

        # Apply the -1 phase to the |11> state
        for i in range(dim):
            # Convert index to binary representation
            binary_state = format(i, f'0{num_qubits}b')

            # Check if both control and target qubits are in the |1> state
            if binary_state[control] == '1' and binary_state[target] == '1':
                full_operator[i, i] = -1  # Apply phase flip

        return full_operator

