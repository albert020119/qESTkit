import random
from collections import Counter

import numpy as np

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


# Define IBM hardware profiles
IBM_PROFILES = {
    "ibm_kyiv": {"gate_error": 0.008, "meas_error": 0.01},
    "ibm_brisbane": {"gate_error": 0.012, "meas_error": 0.02}
}

def run_noisy_simulation(circuit, num_simulations=1000, gate_error_prob=0.01, measurement_error_prob=0.02):
    """
    Simulate a quantum circuit with noise, including gate errors and measurement noise.

    :param circuit: The QuantumCircuit object.
    :param num_simulations: Number of times to simulate the circuit.
    :param gate_error_prob: Probability of introducing random errors in gate operations.
    :param measurement_error_prob: Probability of introducing random errors in measurements.
    :return: A dictionary with counts of each measured outcome.
    """
    # Check if the gate_error_prob matches any IBM profile
    if gate_error_prob == IBM_PROFILES["ibm_kyiv"]["gate_error"]:
        print("Running Hardware Emulation: ibm_kyiv")
    elif gate_error_prob == IBM_PROFILES["ibm_brisbane"]["gate_error"]:
        print("Running Hardware Emulation: ibm_brisbane")

    # Apply all gates in the circuit with gate errors
    state = circuit.state
    for gate in circuit.gates:
        # BULLETPROOF CHECK: Does it have a control qubit, or is the name CNOT/CZ?
        if hasattr(gate, 'control_qubit') or type(gate).__name__.upper() in ['CNOT', 'CZ']:
            print(f"SUCCESS: Bypassing get_operator for {type(gate).__name__}")
            state = gate.apply(state)
        else:
            # Apply Gaussian noise ONLY to single-qubit gates
            if random.random() < gate_error_prob:
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