import numpy as np
from gates import Hadamard
from circuit.Oracle import Oracle
from circuit.Diffuser import Diffuser
from simulator.circuit.circuit_initialization import initialize_superposition


def grover(num_qubits, marked_state):
    """
    Run Grover's algorithm to find the marked state.

    :param num_qubits: Number of qubits in the quantum system.
    :param marked_state: The index of the marked state (integer).
    :return: The final quantum state vector after Grover's iterations.
    """
    num_states = 2**num_qubits

    # Step 1: Initialize the system in |0> state
    state_vector = np.zeros(num_states, dtype=complex)
    state_vector[0] = 1  # |0> state

    # Step 2: Apply Hadamard to all qubits to create superposition
    num_qubits = 6  # Number of qubits
    expected_amplitude = 1 / np.sqrt(2 ** num_qubits)

    # Initialize the state vector
    state_vector = initialize_superposition(num_qubits)

    # Step 3: Create Oracle and Diffuser
    oracle = Oracle(marked_state=marked_state, num_qubits=num_qubits)
    diffuser = Diffuser(num_qubits=num_qubits)

    # Calculate the number of iterations
    num_iterations = int(np.floor(np.pi / 4 * np.sqrt(num_states)) - 0.5)

    # Step 4: Apply Oracle and Diffuser iteratively
    for _ in range(num_iterations):
        # Apply Oracle
        state_vector = oracle.apply(state_vector)

        # Apply Diffuser
        state_vector = diffuser.apply(state_vector)

    # Return the final state vector
    return state_vector


def test_grover():
    num_qubits = 6
    marked_state = 2  # Target state to find (binary |101>)

    # Run Grover's algorithm
    final_state_vector = grover(num_qubits=num_qubits, marked_state=marked_state)

    # The final state should have the highest amplitude at the marked state
    probabilities = np.abs(final_state_vector)**2
    measured_state = np.argmax(probabilities)

    print("Final probabilities:\n", probabilities)
    print(f"Measured state: {measured_state}, Expected: {marked_state}")

    # Assert the result
    assert measured_state == marked_state, (
        f"Grover's algorithm failed. Measured: {measured_state}, Expected: {marked_state}."
    )

    print("Grover's algorithm test passed!")

if __name__ == "__main__":
    test_grover()
