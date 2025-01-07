import numpy as np
from simulator.gates import Ry


def test_ry_gate():
    # Define a single-qubit system in the |0> state
    state_vector = np.array([1, 0], dtype=complex)

    # Define the rotation angle (pi/2 radians)
    theta = np.pi / 2

    # Create a Ry gate with the specified angle, targeting qubit 0
    ry_gate = Ry(theta=theta, qubits=[0])

    # Apply the gate
    new_state_vector = ry_gate.apply(state_vector)

    # Expected result after Ry(pi/2) applied to |0>
    # Ry(pi/2) on |0> gives (sqrt(2)/2)|0> + (sqrt(2)/2)|1>
    expected_state_vector = np.array(
        [np.sqrt(2) / 2, np.sqrt(2) / 2], dtype=complex
    )

    # Assert the result matches the expectation
    assert np.allclose(new_state_vector, expected_state_vector), "Ry gate failed to produce the correct state."


if __name__ == "__main__":
    test_ry_gate()
