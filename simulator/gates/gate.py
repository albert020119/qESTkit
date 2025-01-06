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
        Apply the gate to a quantum state vector.

        :param state_vector: The state vector of the quantum system.
        :return: The modified state vector after applying the gate.
        """
        raise NotImplementedError("The apply method must be implemented in subclasses.")

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