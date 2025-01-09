import numpy as np
from simulator.gates.gate import Gate

class CNOT(Gate):
    def __init__(self, control, target):
        """
        Initialize the CNOT gate.

        :param control: The control qubit index.
        :param target: The target qubit index.
        """
        super().__init__(name="CNOT", qubits=[control, target])
        if control == target:
            raise ValueError("Control and target qubits must be different.")

        # The 4x4 matrix representation of the CNOT gate
        self.matrix = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 1],
                                [0, 0, 1, 0]], dtype=complex)

    def apply(self, state_vector):
        """
        Apply the CNOT gate to a quantum state vector.

        :param state_vector: The state vector of the quantum system.
        :return: The modified state vector after applying the gate.
        """
        if self.qubits is None or len(self.qubits) != 2:
            raise ValueError("CNOT gate requires exactly two target qubits.")

        control, target = self.qubits
        num_qubits = int(np.log2(len(state_vector)))

        if control >= num_qubits or target >= num_qubits:
            raise ValueError("Control or target qubit index exceeds the number of qubits in the state vector.")

        # Ensure the control and target qubits are distinct
        if control == target:
            raise ValueError("Control and target qubits must be different.")

        # Construct the full operator using the Kronecker product
        identity = np.eye(2, dtype=complex)

        operator = self._construct_cnot_operator(control, target, num_qubits)
        return operator @ state_vector

    def _construct_cnot_operator(self, control, target, num_qubits):
        """
        Construct the full matrix representation of the CNOT gate for the given system size.

        :param control: Index of the control qubit.
        :param target: Index of the target qubit.
        :param num_qubits: Total number of qubits in the system.
        :return: Full matrix representation of the CNOT gate.
        """
        # CNOT acts on 2 qubits at a time, create the appropriate projector matrices
        P0 = np.array([[1, 0], [0, 0]], dtype=complex)  # |0><0|
        P1 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1><1|
        X = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli-X

        # Initialize operator as identity
        operator = np.eye(1, dtype=complex)

        for i in range(num_qubits):
            if i == control:
                operator = np.kron(operator, P0)  # |0><0| on control qubit
            elif i == target:
                if control < target:
                    operator = np.kron(operator, np.eye(2, dtype=complex))
                else:
                    operator = np.kron(np.eye(2, dtype=complex), operator)

        # Add the |1><1| part with X applied to the target qubit
        second_term = np.eye(1, dtype=complex)
        for i in range(num_qubits):
            # Add the |1><1| part with X applied to the target qubit
            second_term = np.eye(1, dtype=complex)
            for i in range(num_qubits):
                if i == control:
                    second_term = np.kron(second_term, P1)  # |1><1| on control qubit
                elif i == target:
                    second_term = np.kron(second_term, X)  # X on target qubit
                else:
                    second_term = np.kron(second_term, np.eye(2, dtype=complex))

            return operator + second_term
