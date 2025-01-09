import numpy as np
from simulator.gates import Y


def test_pauli_y_gate():
    # Define a single-qubit system in the |0> state
    state_vector = np.array([1, 0], dtype=complex)

    # Create a Pauli-Y gate targeting qubit 0
    y_gate = Y(qubits=[0])

    # Apply the gate
    new_state_vector = y_gate.apply(state_vector)

    # Expected result is i|1> state
    expected_state_vector = np.array([0, 1j], dtype=complex)

    # Assert the result matches the expectation
    assert np.allclose(new_state_vector, expected_state_vector), "Pauli-Y gate failed to produce the correct state."

    # Test with the |1> state
    state_vector = np.array([0, 1], dtype=complex)
    new_state_vector = y_gate.apply(state_vector)
    # Expected result is -i|0> state
    expected_state_vector = np.array([-1j, 0], dtype=complex)

    # Assert the result matches the expectation
    assert np.allclose(new_state_vector, expected_state_vector), "Pauli-Y gate failed to produce the correct state."


if __name__ == "__main__":
    test_pauli_y_gate()
