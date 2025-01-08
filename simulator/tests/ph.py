import numpy as np
from simulator.gates import Ph

def test_ph_gate():
    # Define a single-qubit system in the |1> state
    state_vector = np.array([0, 1], dtype=complex)

    # Create a Phase gate with a phase shift of π/2, targeting qubit 0
    ph_gate = Ph(delta=np.pi / 2, qubits=[0])

    # Apply the gate
    new_state_vector = ph_gate.apply(state_vector)

    # Expected result after Ph(π/2) applied to |1>
    expected_state_vector = np.array(
        [0, np.exp(1j * np.pi / 2)], dtype=complex
    )

    # Assert the result matches the expectation
    assert np.allclose(new_state_vector, expected_state_vector), "Ph gate failed to produce the correct state."

if __name__ == "__main__":
    test_ph_gate()
