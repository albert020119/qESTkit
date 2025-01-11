from collections import Counter

import numpy as np

class QuantumCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.gates = []
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1  # Initialize to |0...0> state

    def add_gate(self, gate):
        """
        Add a gate to the circuit. Can be a single-qubit or multi-qubit gate.

        :param gate_matrix: Matrix representation of the gate.
        :param target_qubit: Index of the target qubit(s). None for multi-qubit gates.
        """
        if gate.qubits is None:  # Multi-qubit gate
            if gate.matrix.shape != (2**self.num_qubits, 2**self.num_qubits):
                raise ValueError("Multi-qubit gate matrix dimensions do not match circuit size.")
        self.gates.append(gate)

    def apply(self):
        """Apply all gates in the circuit to the quantum state."""
        for gate in self.gates:
            self.state = gate.apply(self.state)

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

    def _apply_multi_qubit_gate(self, gate_matrix):
        """Apply a multi-qubit gate to the state."""
        self.state = gate_matrix @ self.state

    def reset(self):
        """Reset the circuit by clearing all gates and resetting the state."""
        self.gates = []
        self.state = np.zeros(2**self.num_qubits, dtype=complex)
        self.state[0] = 1

    def measure_state(self):
        """
        Measure the quantum state and return the collapsed state and measurement result.

        :return: Tuple (measured_state, measured_basis_state).
        """
        probabilities = np.abs(self.state) ** 2
        measured_basis_state = np.random.choice(len(probabilities), p=probabilities)
        collapsed_state = np.zeros_like(self.state, dtype=complex)
        collapsed_state[measured_basis_state] = 1
        self.state = collapsed_state
        return collapsed_state, measured_basis_state

    def simulate(self, num_measurements=1000):
        """
        Simulate the quantum circuit and measure the output state.

        :param num_measurements: Number of measurements to perform.
        :return: A dictionary containing counts for each basis state.
        """
        # Apply all gates in the circuit to get the final state
        self.apply()

        # Calculate probabilities of each basis state
        probabilities = np.abs(self.state) ** 2

        # Perform measurements based on probabilities
        measurement_results = np.random.choice(
            len(probabilities), size=num_measurements, p=probabilities
        )

        # Count occurrences of each measured state
        counts = Counter(measurement_results)

        # Display the results
        print("Measurement results (counts):", counts)
        print("Empirical probabilities:", {state: count / num_measurements for state, count in counts.items()})

        return counts
