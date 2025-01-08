import numpy as np
from simulator.gates import Identity as I

def test_identity_gate():
    # Define a single-qubit system in the |0> state
    state_vector = np.array([1, 0], dtype=complex)

    # Create an Identity gate, targeting qubit 0
    identity_gate = I(qubits=[0])

    # Apply the gate
    new_state_vector = identity_gate.apply(state_vector)

    # Expected result (Identity gate does not change the state)
    expected_state_vector = state_vector

    # Assert the result matches the expectation
    assert np.allclose(new_state_vector, expected_state_vector), "Identity gate failed to preserve the state."
    print("Identity gate test passed!")

if __name__ == "__main__":
    test_identity_gate()
