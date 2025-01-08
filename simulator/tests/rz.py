import numpy as np
from simulator.gates import Rz

def test_rz_gate():
    # Define a single-qubit system in the |1> state
    state_vector = np.array([0, 1], dtype=complex)

    # Create an Rz gate with a rotation of π/4, targeting qubit 0
    rz_gate = Rz(theta=np.pi / 4, qubits=[0])

    # Apply the gate
    new_state_vector = rz_gate.apply(state_vector)

    # Expected result after Rz(π/4) applied to |1>
    expected_state_vector = np.array(
        [0, np.exp(1j * np.pi / 8)], dtype=complex
    )

    # Assert the result matches the expectation
    assert np.allclose(new_state_vector, expected_state_vector), "Rz gate failed to produce the correct state."

if __name__ == "__main__":
    test_rz_gate()
