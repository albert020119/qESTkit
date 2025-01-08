import numpy as np

class QuantumCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.gates = []
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1  # Initialize to |0...0> state

    def add_gate(self, gate_matrix, target_qubit):
        """Add a single-qubit gate to the circuit."""
        self.gates.append(('single', gate_matrix, [target_qubit]))

    def add_cnot(self, control_qubit, target_qubit):
        """Add a CNOT (controlled-NOT) gate to the circuit."""
        self.gates.append(('cnot', None, [control_qubit, target_qubit]))

    def apply(self):
        """Apply all gates in the circuit to the quantum state."""
        for gate_type, gate_matrix, target_qubits in self.gates:
            if gate_type == 'single':
                self._apply_single_qubit_gate(gate_matrix, target_qubits[0])
            elif gate_type == 'cnot':
                self._apply_cnot(target_qubits[0], target_qubits[1])

    def _apply_single_qubit_gate(self, gate_matrix, target_qubit):
        """Apply a single-qubit gate to the state."""
        identity = np.eye(2, dtype=complex)
        operator = 1
        for i in range(self.num_qubits):
            if i == target_qubit:
                operator = np.kron(operator, gate_matrix)
            else:
                operator = np.kron(operator, identity)
        self.state = operator @ self.state

    def _apply_cnot(self, control_qubit, target_qubit):
        """Apply a CNOT gate to the state."""
        size = 2**self.num_qubits
        cnot_matrix = np.eye(size, dtype=complex)
        for i in range(size):
            if (i >> control_qubit) & 1:
                j = i ^ (1 << target_qubit)
                cnot_matrix[i, i] = 0
                cnot_matrix[j, j] = 0
                cnot_matrix[i, j] = 1
                cnot_matrix[j, i] = 1
        self.state = cnot_matrix @ self.state

    def reset(self):
        """Reset the circuit by clearing all gates and resetting the state."""
        self.gates = []
        self.state = np.zeros(2**self.num_qubits, dtype=complex)
        self.state[0] = 1
