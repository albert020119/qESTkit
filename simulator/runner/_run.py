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
    "ibm_kyiv": {
        "gate_error": 0.008,
        "meas_error": 0.01,
        "single_qubit_error": 0.001,
        "cnot_error": 0.01,
        "t1_decay_prob": 0.03,
        "thermal_prob": 0.002
    },
    "ibm_brisbane": {
        "gate_error": 0.012,
        "meas_error": 0.02,
        "single_qubit_error": 0.002,  # made the probabilities bigger cause its usually noisier than kyiv
        "cnot_error": 0.02,           
        "t1_decay_prob": 0.04,       
        "thermal_prob": 0.005         
    }
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
    # Intercept IBM profile if gate_error_prob matches
    profile = None
    for name, params in IBM_PROFILES.items():
        if gate_error_prob == params["gate_error"]:
            profile = params
            print(f"Running Hardware Emulation: {name}")
            break

    # Default to basic noise if no profile matched
    single_qubit_error = profile["single_qubit_error"] if profile else gate_error_prob
    cnot_error = profile["cnot_error"] if profile else gate_error_prob
    t1_decay_prob = profile["t1_decay_prob"] if profile else 0.0
    thermal_prob = profile["thermal_prob"] if profile else 0.0

    # Apply all gates in the circuit with gate errors
    state = circuit.state
    for gate in circuit.gates:
        if hasattr(gate, 'control_qubit') or type(gate).__name__.upper() in ['CNOT', 'CZ']:
            # Apply ideal gate
            state = gate.apply(state)
            # Apply CNOT error
            if random.random() < cnot_error:
                noise = np.random.normal(0, 0.01, state.shape) + 1j * np.random.normal(0, 0.01, state.shape)
                state += noise
        else:
            # Apply single-qubit gate error
            if random.random() < single_qubit_error:
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

    # Perform measurements with advanced or basic noise
    measurement_results = []
    for _ in range(num_simulations):
        # Simulate a measurement
        true_state = np.random.choice(len(probabilities), p=probabilities)

        if profile:
            # Advanced noise: T1 decay and thermal excitation
            true_state_binary = bin(true_state)[2:].zfill(circuit.num_qubits)
            noisy_state_binary = ""
            for bit in true_state_binary:
                if bit == '1':
                    # T1 decay: |1> -> |0>
                    noisy_state_binary += '0' if random.random() < t1_decay_prob else '1'
                elif bit == '0':
                    # Thermal excitation: |0> -> |1>
                    noisy_state_binary += '1' if random.random() < thermal_prob else '0'
            noisy_state = int(noisy_state_binary, 2)
        else:
            # Basic noise: Random flip
            if random.random() < measurement_error_prob:
                noisy_state = random.randint(0, len(probabilities) - 1)
            else:
                noisy_state = true_state

        measurement_results.append(noisy_state)

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