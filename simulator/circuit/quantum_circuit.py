class QuantumCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.gates = []  # Store gates as a list of tuples (gate_matrix, target_qubits)

    def add_gate(self, gate_matrix, target_qubit):
        """Add a single-qubit gate to the circuit."""
        self.gates.append(('single', gate_matrix, [target_qubit]))

    def add_cnot(self, control_qubit, target_qubit):
        """Add a CNOT (controlled-NOT) gate to the circuit."""
        self.gates.append(('cnot', None, [control_qubit, target_qubit]))

    def apply(self, quantum_register):
        """Apply all gates in the circuit to the given QuantumRegister."""
        for gate_type, gate_matrix, target_qubits in self.gates:
            if gate_type == 'single':
                quantum_register.apply_gate(gate_matrix, target_qubits[0])
            elif gate_type == 'cnot':
                quantum_register.apply_cnot(target_qubits[0], target_qubits[1])

    def reset(self):
        """Reset the circuit by clearing all gates."""
        self.gates = []
