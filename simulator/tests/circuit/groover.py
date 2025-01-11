import random
from collections import Counter

import numpy as np
from simulator.gates import Hadamard, X, CZ
from simulator.circuit.quantum_circuit import QuantumCircuit

def run_simulation(circuit, num_simulations=100):
    """
    Simulate a quantum circuit multiple times, returning counts of measurement outcomes
    and printing raw probabilities after applying the circuit.

    :param circuit: The QuantumCircuit object.
    :param num_simulations: Number of times to simulate the circuit.
    :return: A dictionary with counts of each measured outcome.
    """
    # Apply all gates in the circuit to compute the final state
    circuit.apply()

    # Calculate probabilities of each basis state
    probabilities = np.abs(circuit.state) ** 2

    # Print raw probabilities for all basis states
    print("\nRaw State Probabilities (Unrounded):")
    for state, prob in enumerate(probabilities):
        state_str = f"|{bin(state)[2:].zfill(circuit.num_qubits)}>"
        print(f"{state_str}: {prob}")

    # Verify probability normalization
    total_probability = np.sum(probabilities)
    print(f"\nTotal Probability: {total_probability}")
    if not np.isclose(total_probability, 1.0):
        raise ValueError("Probabilities are not normalized. Check the gate implementations.")

    # Perform measurements
    measurement_results = np.random.choice(
        len(probabilities), size=num_simulations, p=probabilities
    )

    # Count the occurrences of each measurement outcome
    counts = Counter(measurement_results)

    # Format results
    formatted_counts = {
        f"|{bin(state)[2:].zfill(circuit.num_qubits)}>": count
        for state, count in counts.items()
    }

    # Print the measurement results
    print("\nMeasurement Results (Counts):")
    for state, count in formatted_counts.items():
        print(f"{state}: {count}")

    return formatted_counts


def run_noisy_simulation(circuit, num_simulations=1000, gate_error_prob=0.01, measurement_error_prob=0.02):
    """
    Simulate a quantum circuit with noise, including gate errors and measurement noise.

    :param circuit: The QuantumCircuit object.
    :param num_simulations: Number of times to simulate the circuit.
    :param gate_error_prob: Probability of introducing random errors in gate operations.
    :param measurement_error_prob: Probability of introducing random errors in measurements.
    :return: A dictionary with counts of each measured outcome.
    """
    # Apply all gates in the circuit with gate errors
    state = circuit.state
    for gate in circuit.gates:
        if random.random() < gate_error_prob:
            # Introduce random gate error: Add a small random perturbation to the operator
            noise = np.random.normal(0, 0.01, gate.get_operator(circuit.num_qubits).shape)
            noisy_operator = gate.get_operator(circuit.num_qubits) + noise
            state = noisy_operator @ state
        else:
            state = gate.get_operator(circuit.num_qubits) @ state

    # Calculate probabilities of each basis state
    probabilities = np.abs(state) ** 2

    # Verify probability normalization
    total_probability = np.sum(probabilities)
    if not np.isclose(total_probability, 1.0):
        probabilities /= total_probability  # Normalize in case of noise

    # Perform measurements with measurement noise
    measurement_results = []
    for _ in range(num_simulations):
        # Simulate a noisy measurement
        true_state = np.random.choice(len(probabilities), p=probabilities)
        if random.random() < measurement_error_prob:
            # Flip to a random state due to measurement error
            noisy_state = random.randint(0, len(probabilities) - 1)
            measurement_results.append(noisy_state)
        else:
            measurement_results.append(true_state)

    # Count occurrences of each measurement outcome
    counts = Counter(measurement_results)

    # Format results
    formatted_counts = {
        f"|{bin(state)[2:].zfill(circuit.num_qubits)}>": count
        for state, count in counts.items()
    }

    # Print the noisy results
    print("\nNoisy Measurement Results (Counts):")
    for state, count in formatted_counts.items():
        print(f"{state}: {count}")

    return formatted_counts

def grover_3_qubits_with_simulation():
    # Create a quantum circuit with 3 qubits (2 data qubits + 1 auxiliary qubit)
    qc = QuantumCircuit(num_qubits=2)

    # Step 1: Apply Hadamard gates to the data qubits to create a superposition
    qc.add_gate(Hadamard(qubits=[0]))
    qc.add_gate(Hadamard(qubits=[1]))

    # Step 2: Oracle - mark the |11> state by applying a phase flip
    qc.add_gate(CZ(qubits=[0, 1]))

    # Step 3: Diffusion operator
    qc.add_gate(Hadamard(qubits=[0]))
    qc.add_gate(Hadamard(qubits=[1]))
    qc.add_gate(X(qubits=[0]))
    qc.add_gate(X(qubits=[1]))
    qc.add_gate(CZ(qubits=[0, 1]))
    qc.add_gate(X(qubits=[0]))
    qc.add_gate(X(qubits=[1]))
    qc.add_gate(Hadamard(qubits=[0]))
    qc.add_gate(Hadamard(qubits=[1]))

    # Print the circuit structure (if needed for debugging)
    print(qc)

    results = run_simulation(qc, num_simulations=1)
    run_noisy_simulation(qc, num_simulations=1000, gate_error_prob=0.05, measurement_error_prob=0.1)


# Run the Grover's algorithm
grover_3_qubits_with_simulation()
