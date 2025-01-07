import numpy as np
from simulator.gates import Rx


def test_rx_gate():
    # Define a single-qubit system in the |0> state
    state_vector = np.array([1, 0], dtype=complex)

    # Define the rotation angle (pi/2 radians)
    theta = np.pi / 2

    # Create a Rx gate with the specified angle, targeting qubit 0
    rx_gate = Rx(theta=theta, qubits=[0])

    # Apply the gate
    new_state_vector = rx_gate.apply(state_vector)

    # Expected result after Rx(pi/2) applied to |0>
    # Rx(pi/2) on |0> gives (sqrt(2)/2)|0> - i(sqrt(2)/2)|1>
    expected_state_vector = np.array(
        [np.sqrt(2) / 2, -1j * np.sqrt(2) / 2], dtype=complex
    )

    # Assert the result matches the expectation
    assert np.allclose(new_state_vector, expected_state_vector), "Rx gate failed to produce the correct state."


if __name__ == "__main__":
    test_rx_gate()
