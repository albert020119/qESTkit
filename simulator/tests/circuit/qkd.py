import random

from simulator.circuit.quantum_circuit import QuantumCircuit
from simulator.gates import X, Hadamard


def prepare_qubits(num_bits):
    """
    Step 1: Alice prepares random bits and bases.
    :return: Tuple (alice_bits, alice_bases, alice_circuits)
    """
    alice_bits = [random.randint(0, 1) for _ in range(num_bits)]
    alice_bases = [random.choice(['Z', 'X']) for _ in range(num_bits)]
    alice_circuits = []

    for i in range(num_bits):
        qc = QuantumCircuit(num_qubits=1)
        if alice_bits[i] == 1:
            qc.add_gate(X(qubits=[0]))  # Prepare |1> state
        if alice_bases[i] == 'X':
            qc.add_gate(Hadamard(qubits=[0]))  # Apply Hadamard for X-basis
        alice_circuits.append(qc)

    return alice_bits, alice_bases, alice_circuits

def measure_qubits(alice_circuits, num_bits):
    """
    Step 2: Bob measures qubits with random bases.
    :param num_bits: number of bits to measure.
    :param alice_circuits: Circuits prepared by Alice.
    :return: Tuple (bob_bases, bob_results)
    """
    bob_bases = [random.choice(['Z', 'X']) for _ in range(num_bits)]
    bob_results = []

    for i, qc in enumerate(alice_circuits):
        if bob_bases[i] == 'X':
            qc.add_gate(Hadamard(qubits=[0]))  # Change basis to X if needed
        qc.apply()
        _, measured_bit = qc.measure_state()
        bob_results.append(measured_bit)

    return bob_bases, bob_results

def compare_bases(alice_bases, bob_bases, alice_bits, bob_results, num_bits):
    """
    Step 3: Alice and Bob compare bases.
    :return: The shared key.
    """
    shared_key = [
        alice_bits[i]
        for i in range(num_bits)
        if alice_bases[i] == bob_bases[i]
    ]
    return shared_key

"""
Run the full QKD simulation using your library.
"""
# Step 1: Alice prepares qubits
num_bits = 10
alice_bits, alice_bases, alice_circuits = prepare_qubits(num_bits)

# Step 2: Bob measures qubits
bob_bases, bob_results = measure_qubits(alice_circuits, num_bits)

# Step 3: Compare bases and extract the shared key
shared_key = compare_bases(alice_bases, bob_bases, alice_bits, bob_results, num_bits)

print("Alice's Bits: ", alice_bits)
print("Alice's Bases: ", alice_bases)
print("Bob's Bases:   ", bob_bases)
print("Bob's Results: ", bob_results)
print("Shared Key:    ", shared_key)