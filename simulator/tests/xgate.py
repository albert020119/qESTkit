import numpy as np

from simulator.gates import X


def test_pauli_x_gate():
    # Define a single-qubit system in the |0> state
    state_vector = np.array([1, 0], dtype=complex)

    # Create a Pauli-X gate targeting qubit 0
    x_gate = X(qubits=[0])

    # Apply the gate
    new_state_vector = x_gate.apply(state_vector)

    # Expected result is the |1> state
    expected_state_vector = np.array([0, 1], dtype=complex)
    
    # Assert the result matches the expectation
    assert np.allclose(new_state_vector, expected_state_vector), "Pauli-X gate failed to produce the correct state."


if __name__ == "__main__":
    test_pauli_x_gate()