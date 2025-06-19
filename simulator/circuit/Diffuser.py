import numpy as np

class Diffuser:
    def __init__(self, num_qubits):
        """
        Initialize the Diffuser for Grover's algorithm.

        :param num_qubits: Number of qubits in the quantum system.
        """
        self.num_qubits = num_qubits
        self.num_states = 2**num_qubits

        # Create the diffusion operator
        # Start with the all-ones matrix divided by the number of states
        self.mean_operator = 2 * np.full((self.num_states, self.num_states), 1 / self.num_states, dtype=complex)

        # Subtract the identity matrix
        self.matrix = self.mean_operator - np.eye(self.num_states, dtype=complex)

    def apply(self, state_vector):
        """
        Apply the Diffuser to a state vector.

        :param state_vector: The quantum state vector.
        :return: The modified state vector after applying the Diffuser.
        """
        return self.matrix @ state_vector
