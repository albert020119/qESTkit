import numpy as np
from simulator.gates import T


def test_t_gate():
    # Define a single-qubit system in the |0> state
    state_vector = np.array([1, 0], dtype=complex)

    # Create a T gate targeting qubit 0
    t_gate = T(qubits=[0])

    # Apply the gate
    new_state_vector = t_gate.apply(state_vector)

    # Expected result after T applied to |0>
    # T|0> = |0>, as the T gate does not affect the |0> component
    expected_state_vector = np.array([1, 0], dtype=complex)

    # Assert the result matches the expectation
    assert np.allclose(new_state_vector, expected_state_vector), "T gate failed to produce the correct state on |0>."

    # Test on |1> state
    state_vector = np.array([0, 1], dtype=complex)
    new_state_vector = t_gate.apply(state_vector)

    # Expected result after T applied to |1>
    # T|1> = e^{iπ/4}|1>
    expected_state_vector = np.array([0, np.exp(1j * np.pi / 4)], dtype=complex)

    # Assert the result matches the expectation
    assert np.allclose(new_state_vector, expected_state_vector), "T gate failed to produce the correct state on |1>."


if __name__ == "__main__":
    test_t_gate()
