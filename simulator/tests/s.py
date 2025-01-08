import numpy as np
from simulator.gates import S

def test_s_gate():
    # Define a single-qubit system in the |1> state
    state_vector = np.array([0, 1], dtype=complex)

    # Create an S gate, targeting qubit 0
    s_gate = S(qubits=[0])

    # Apply the gate
    new_state_vector = s_gate.apply(state_vector)

    # Expected result after S applied to |1>
    expected_state_vector = np.array(
        [0, 1j], dtype=complex
    )

    # Assert the result matches the expectation
    assert np.allclose(new_state_vector, expected_state_vector), "S gate failed to produce the correct state."

if __name__ == "__main__":
    test_s_gate()
