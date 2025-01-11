import numpy as np

class Gate:
    def __init__(self, name, qubits=None):
        """
        Base class for quantum gates.

        :param name: Name of the gate (e.g., 'X', 'H').
        :param qubits: List of qubit indices this gate acts on.
        """
        self.name = name
        self.qubits = qubits if qubits is not None else []
        self.matrix = None  # Should be defined in subclasses

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
        if target_qubit == 0:
            operator = self.matrix
        else:
            operator = identity
        for i in range(1, num_qubits):
            if i == target_qubit:
                operator = np.kron(operator, self.matrix)
            else:
                operator = np.kron(operator, identity)

        return operator @ state_vector

    def validate(self, num_qubits):
        """
        Validate the gate's target qubits against the total number of qubits.

        :param num_qubits: Total number of qubits in the quantum system.
        :raises ValueError: If any target qubit index is invalid.
        """
        if any(q >= num_qubits or q < 0 for q in self.qubits):
            raise ValueError(f"Invalid qubit indices {self.qubits} for a system with {num_qubits} qubits.")

    def get_operator(self, num_qubits):
        """
        Construct the full operator for the quantum system.

        :param num_qubits: Total number of qubits in the quantum system.
        :return: A matrix representing the gate operator for the full system.
        """
        if self.matrix is None:
            raise ValueError("Gate matrix is not defined.")

        identity = np.eye(2, dtype=complex)
        operator = self.matrix

        for i in range(num_qubits):
            if i == self.qubits[0]:
                continue
            operator = np.kron(identity, operator)

        return operator

    def __repr__(self):
        return f"Gate(name={self.name}, qubits={self.qubits})"