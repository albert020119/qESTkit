from collections import Counter
import matplotlib.pyplot as plt
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

    def run_circuit(self):
        """
        Run the quantum circuit and visualize the state probabilities.
        """
        # Step 1: Apply all gates
        self.apply()

        # Step 2: Calculate probabilities of each basis state
        probabilities = np.abs(self.state) ** 2

        # Step 3: Generate the labels for the states
        num_states = len(probabilities)
        state_labels = [format(i, f'0{self.num_qubits}b') for i in range(num_states)]

        # Step 4: Plot the probabilities
        plt.bar(state_labels, probabilities)
        plt.xlabel("Basis States")
        plt.ylabel("Probability")
        plt.title("Quantum State Probabilities")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def draw(self):
        """
        Visualize the quantum circuit, showing qubits and gates.
        """
        fig_width = max(6, len(self.gates) * 1.5)
        fig_height = max(2, self.num_qubits * 1.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        for i in range(self.num_qubits):
            ax.hlines(y=self.num_qubits - 1 - i, xmin=0, xmax=len(self.gates), color='gray', linewidth=1.5, linestyle='--', label=f'q[{i}]' if i == 0 else "")

        for step, gate in enumerate(self.gates):
            if len(gate.qubits) == 1:
                qubit = self.num_qubits - 1 - gate.qubits[0]
                ax.text(step + 0.5, qubit, gate.name, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='deepskyblue', edgecolor='black', alpha=0.8), fontsize=12)
            else:
                min_qubit, max_qubit = min(gate.qubits), max(gate.qubits)
                control_qubit = self.num_qubits - 1 - gate.qubits[0]
                target_qubit = self.num_qubits - 1 - gate.qubits[1]

                ax.vlines(x=step + 0.5, ymin=target_qubit, ymax=control_qubit, color='black', linestyle='dotted', linewidth=1.2)

                ax.plot(step + 0.5, control_qubit, 'o', color='black', markersize=8)

                ax.text(step + 0.5, target_qubit, gate.name, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='deepskyblue', edgecolor='black', alpha=0.8))

        ax.set_xlim(0, len(self.gates))
        ax.set_ylim(-1, self.num_qubits)
        ax.set_yticks(range(self.num_qubits))
        ax.set_yticklabels([f'q[{i}]' for i in range(self.num_qubits - 1, -1, -1)])
        ax.set_xticks(range(len(self.gates) + 1))
        ax.set_xticklabels([str(i) for i in range(len(self.gates) + 1)])
        ax.set_xlabel("Gate Steps")
        ax.set_title("Quantum Circuit", fontsize=14, fontweight='bold')
        ax.grid(False)

        ax.set_facecolor('#f9f9f9')

        plt.subplots_adjust(left=0.1)

        plt.tight_layout()
        plt.show()

