import numpy as np

class QuantumRegister:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state_vector = np.zeros(2**num_qubits, dtype=complex)  # Create state vector
        self.state_vector[0] = 1  # Initialize to |0...0> state

    def apply_gate(self, gate_matrix, target_qubit):
        # Expand the gate to act on the entire register by creating an identity
        full_gate = np.eye(1)
        for i in range(self.num_qubits):
            if i == target_qubit:
                full_gate = np.kron(full_gate, gate_matrix)  # Apply gate to target qubit
            else:
                full_gate = np.kron(full_gate, np.eye(2))    # Apply identity to other qubits
        self.state_vector = np.dot(full_gate, self.state_vector)  # Update state vector

    def measure(self):
        # Probabilities of measuring each state
        probabilities = np.abs(self.state_vector) ** 2
        # Choose a state based on these probabilities
        measured_state = np.random.choice(2**self.num_qubits, p=probabilities)
        # Reset the state vector to the measured state
        new_state_vector = np.zeros(2**self.num_qubits, dtype=complex)
        new_state_vector[measured_state] = 1
        self.state_vector = new_state_vector
        # Return the measured state in binary representation
        return format(measured_state, f'0{self.num_qubits}b')
