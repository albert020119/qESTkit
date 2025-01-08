import numpy as np
from simulator.gates import Hadamard


def test_hadamard_gate():
    # Define a single-qubit system in the |0> state
    state_vector = np.array([1, 0], dtype=complex)

    # Create a Hadamard gate targeting qubit 0
    hadamard_gate = Hadamard(qubits=[0])

    # Apply the gate
    new_state_vector = hadamard_gate.apply(state_vector)

    # Expected result after H applied to |0>
    # H|0> = (1/sqrt(2))|0> + (1/sqrt(2))|1>
    expected_state_vector = np.array(
        [1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex
    )

    # Assert the result matches the expectation
    assert np.allclose(new_state_vector, expected_state_vector), "Hadamard gate failed to produce the correct state."

    # Test on |1> state
    state_vector = np.array([0, 1], dtype=complex)
    new_state_vector = hadamard_gate.apply(state_vector)

    # Expected result after H applied to |1>
    # H|1> = (1/sqrt(2))|0> - (1/sqrt(2))|1>
    expected_state_vector = np.array(
        [1 / np.sqrt(2), -1 / np.sqrt(2)], dtype=complex
    )

    # Assert the result matches the expectation
    assert np.allclose(new_state_vector, expected_state_vector), "Hadamard gate failed to produce the correct state on |1>."


if __name__ == "__main__":
    test_hadamard_gate()
