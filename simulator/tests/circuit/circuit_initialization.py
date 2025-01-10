import numpy as np
from simulator.gates import Hadamard

def initialize_superposition(num_qubits):
    """
    Initialize a quantum system of `num_qubits` in superposition.

    :param num_qubits: Number of qubits in the system.
    :return: State vector of the system in superposition.
    """
    # Start with the |0...0> state (all qubits in |0>)
    state_vector = np.zeros(2**num_qubits, dtype=complex)
    state_vector[0] = 1  # Set the |0...0> state to amplitude 1

    # Apply the Hadamard gate to each qubit
    for qubit in range(num_qubits):
        hadamard_gate = Hadamard(qubits=[qubit])
        state_vector = hadamard_gate.apply(state_vector)

    return state_vector


def test_initialize_superposition():
    """
    Test the initialization of a quantum system in superposition.
    """
    num_qubits = 6  # Example: 3 qubits
    expected_amplitude = 1 / np.sqrt(2**num_qubits)

    # Initialize the state vector
    state_vector = initialize_superposition(num_qubits)

    # Verify the amplitudes of all states are equal
    expected_state_vector = np.array(
        [expected_amplitude] * (2**num_qubits), dtype=complex
    )

    # Assert the result matches the expectation
    assert np.allclose(state_vector, expected_state_vector), (
        f"Initialization failed. Expected {expected_state_vector}, got {state_vector}"
    )

    print(f"Test passed for {num_qubits} qubits. State is \n{state_vector}")

if __name__ == "__main__":
    test_initialize_superposition()
