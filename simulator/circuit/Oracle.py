import numpy as np

class Oracle:
    def __init__(self, marked_state, num_qubits):
        """
        Initialize the Oracle.

        :param marked_state: The marked state (integer) to flip the phase for.
        :param num_qubits: The number of qubits in the system.
        """
        self.marked_state = marked_state
        self.num_qubits = num_qubits
        self.num_states = 2**num_qubits

        # Construct the oracle matrix
        self.matrix = np.eye(self.num_states, dtype=complex)
        self.matrix[marked_state, marked_state] = -1  # Flip the phase for the marked state

    def apply(self, state_vector):
        """
        Apply the Oracle to a state vector.

        :param state_vector: The quantum state vector.
        :return: The modified state vector after applying the Oracle.
        """
        return self.matrix @ state_vector


def test_oracle():
    # Define a 2-qubit system (N = 4)
    num_qubits = 6
    num_states = 2**num_qubits

    # Initialize a state vector (equal superposition of all states)
    state_vector = np.ones(num_states, dtype=complex) / np.sqrt(num_states)

    # Create the Oracle that marks the state |10> (decimal 2)
    marked_state = 10
    oracle = Oracle(marked_state=marked_state, num_qubits=num_qubits)

    # Apply the Oracle
    new_state_vector = oracle.apply(state_vector)

    # Check the phase of the marked state |10>
    expected_state_vector = state_vector.copy()
    expected_state_vector[marked_state] *= -1  # Flip the phase of the marked state

    # Assert the result matches the expectation
    assert np.allclose(new_state_vector, expected_state_vector), (
        f"Oracle test failed. Expected {expected_state_vector}, got {new_state_vector}."
    )

    print(f"Oracle test passed! Vector state is:\n {new_state_vector}")

if __name__ == "__main__":
    test_oracle()