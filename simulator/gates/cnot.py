import numpy as np
from .gate import Gate

class CNOT(Gate):
    def __init__(self, control_qubit, target_qubit):
        """
        Initialize the CNOT (Controlled-NOT) gate.

        :param control_qubit: Index of the control qubit.
        :param target_qubit: Index of the target qubit.
        """
        super().__init__(name='CNOT', qubits=[control_qubit, target_qubit])

        # The CNOT gate matrix for 2 qubits
        self.matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)

    def apply(self, state_vector):
        """
        Apply the CNOT gate to the given quantum state vector.

        :param state_vector: The state vector of the quantum system.
        :return: The modified state vector after applying the CNOT gate.
        """
        if len(self.qubits) != 2:
            raise ValueError("CNOT gate requires exactly two qubits: control and target.")

        control_qubit, target_qubit = self.qubits
        num_qubits = int(np.log2(len(state_vector)))

        if control_qubit >= num_qubits or target_qubit >= num_qubits:
            raise ValueError("Control or target qubit index exceeds the number of qubits in the state vector.")

        # Generate the full operator for the system
        operator = self.get_operator(num_qubits)

        # Apply the operator to the state vector
        return operator @ state_vector

    def get_operator(self, num_qubits):
        """
        Construct the full operator for the quantum system.

        :param num_qubits: Total number of qubits in the quantum system.
        :return: A matrix representing the CNOT operator for the full system.
        """
        if self.matrix is None:
            raise ValueError("Gate matrix is not defined.")

        # Start with identity
        identity = np.eye(2, dtype=complex)

        # Build the full operator using Kronecker products
        operator = 1
        for i in range(num_qubits):
            if i == self.qubits[0]:  # Control qubit
                continue
            elif i == self.qubits[1]:  # Target qubit
                operator = np.kron(operator, self.matrix)
            else:
                operator = np.kron(operator, identity)

        return operator

    def __repr__(self):
        return f"CNOT(control_qubit={self.qubits[0]}, target_qubit={self.qubits[1]})"
