import random

from simulator.circuit.quantum_circuit import QuantumCircuit
from simulator.gates import X, Hadamard


class QKDWithEavesdropping:
    def __init__(self, num_bits):
        """
        Initialize the QKD simulation with an eavesdropper (Eve).
        :param num_bits: Number of qubits to simulate.
        """
        self.num_bits = num_bits

    def prepare_qubits(self):
        """
        Step 1: Alice prepares random bits and bases.
        :return: Tuple (alice_bits, alice_bases, alice_circuits)
        """
        alice_bits = [random.randint(0, 1) for _ in range(self.num_bits)]
        alice_bases = [random.choice(['Z', 'X']) for _ in range(self.num_bits)]
        alice_circuits = []

        for i in range(self.num_bits):
            qc = QuantumCircuit(num_qubits=1)
            if alice_bits[i] == 1:
                qc.add_gate(X(qubits=[0]))  # Prepare |1> state
            if alice_bases[i] == 'X':
                qc.add_gate(Hadamard(qubits=[0]))  # Apply Hadamard for X-basis
            alice_circuits.append(qc)

        return alice_bits, alice_bases, alice_circuits

    def eavesdrop(self, alice_circuits):
        """
        Step 2: Eve intercepts and measures qubits with random bases.
        :param alice_circuits: Circuits prepared by Alice.
        :return: Eve-modified circuits sent to Bob.
        """
        eve_bases = [random.choice(['Z', 'X']) for _ in range(self.num_bits)]
        modified_circuits = []

        for i, qc in enumerate(alice_circuits):
            # Eve measures in a random basis
            if eve_bases[i] == 'X':
                qc.add_gate(Hadamard(qubits=[0]))  # Change to X-basis
            qc.apply()
            _, measured_bit = qc.measure_state()  # Eve measures the qubit

            # Eve resends the qubit (modified state) to Bob
            new_qc = QuantumCircuit(num_qubits=1)
            if measured_bit == 1:
                new_qc.add_gate(X(qubits=[0]))  # Prepare |1> state
            if eve_bases[i] == 'X':
                new_qc.add_gate(Hadamard(qubits=[0]))  # Apply Hadamard if in X-basis
            modified_circuits.append(new_qc)

        return modified_circuits

    def measure_qubits(self, circuits_sent_to_bob):
        """
        Step 3: Bob measures qubits with random bases.
        :param circuits_sent_to_bob: Circuits received by Bob.
        :return: Tuple (bob_bases, bob_results)
        """
        bob_bases = [random.choice(['Z', 'X']) for _ in range(self.num_bits)]
        bob_results = []

        for i, qc in enumerate(circuits_sent_to_bob):
            if bob_bases[i] == 'X':
                qc.add_gate(Hadamard(qubits=[0]))  # Change to X-basis if needed
            qc.apply()
            _, measured_bit = qc.measure_state()
            bob_results.append(measured_bit)

        return bob_bases, bob_results

    def compare_bases(self, alice_bases, bob_bases, alice_bits, bob_results):
        """
        Step 4: Alice and Bob compare bases and compute the error rate on a random subset of matching bits.
        :return: The shared key, error rate, and the test subset used for error detection.
        """
        matching_indices = [
            i for i in range(self.num_bits) if alice_bases[i] == bob_bases[i]
        ]
        shared_key = [alice_bits[i] for i in matching_indices]
        bob_key = [bob_results[i] for i in matching_indices]

        if matching_indices:
            # Select a random subset of matching indices for error rate calculation
            subset_size = max(1, len(matching_indices) // 4)  # Use 25% of the matching bits
            subset_indices = random.sample(range(len(matching_indices)), subset_size)

            # Retrieve the indices in terms of the original bits
            test_indices = [matching_indices[i] for i in subset_indices]

            test_alice_bits = [shared_key[i] for i in subset_indices]
            test_bob_bits = [bob_key[i] for i in subset_indices]

            errors = sum(1 for i in range(subset_size) if test_alice_bits[i] != test_bob_bits[i])
            error_rate = errors / subset_size if subset_size else 0

            # Remove the test bits from the shared key for final usage
            final_shared_key = [
                shared_key[i] for i in range(len(shared_key)) if i not in subset_indices
            ]
        else:
            final_shared_key = []
            error_rate = 0
            test_indices = []

        print(f"Tested Indices (Original): {test_indices}")
        return final_shared_key, bob_key, error_rate, test_indices

    def run(self):
        """
        Run the full QKD simulation with Eve.
        """
        # Step 1: Alice prepares qubits
        alice_bits, alice_bases, alice_circuits = self.prepare_qubits()

        # Step 2: Eve intercepts and modifies qubits
        circuits_sent_to_bob = self.eavesdrop(alice_circuits)

        # Step 3: Bob measures the qubits
        bob_bases, bob_results = self.measure_qubits(circuits_sent_to_bob)

        # Step 4: Compare bases and extract the shared key
        shared_key, bob_key, error_rate, test_indices = self.compare_bases(
            alice_bases, bob_bases, alice_bits, bob_results
        )

        print("Alice's Bits:   ", alice_bits)
        print("Alice's Bases:  ", alice_bases)
        print("Bob's Bases:    ", bob_bases)
        print("Bob's Results:  ", bob_results)
        print("Shared Key:     ", shared_key)
        print("Bob's Key:      ", bob_key)
        print(f"Tested Indices:     ", test_indices)

        print(f"Error Rate:     {error_rate * 100:.2f}%")

        if error_rate > 0.1:  # Arbitrary threshold for detecting Eve
            print("Eavesdropping detected!")
        else:
            print("No eavesdropping detected.")

        return shared_key


# Example Usage
qkd_with_eve = QKDWithEavesdropping(num_bits=16)  # Simulate with 16 qubits
qkd_with_eve.run()