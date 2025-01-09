import numpy as np
from simulator.gates import Z

def test_pauli_z_gate():
    # Test with the |0⟩ state
    state_vector = np.array([1, 0], dtype=complex)

    # Create a Z gate targeting qubit 0
    z_gate = Z(qubits=[0])

    # Apply the Z gate
    new_state_vector = z_gate.apply(state_vector)

    # The expected result is the same |0⟩ state
    expected_state_vector = np.array([1, 0], dtype=complex)

    # Assert the result matches the expectation
    assert np.allclose(new_state_vector, expected_state_vector), "Pauli-Z gate failed for |0⟩ state."

    # Test with the |1⟩ state
    state_vector = np.array([0, 1], dtype=complex)
    new_state_vector = z_gate.apply(state_vector)

    # The expected result is -|1⟩ state
    expected_state_vector = np.array([0, -1], dtype=complex)

    # Assert the result matches the expectation
    assert np.allclose(new_state_vector, expected_state_vector), "Pauli-Z gate failed for |1⟩ state."


if __name__ == "__main__":
    test_pauli_z_gate()
