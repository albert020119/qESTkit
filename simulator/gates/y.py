import numpy as np

from simulator.gates.gate import Gate


class Y(Gate):
    def __init__(self, qubits):
        """
        Initialize the Y gate.

        :param qubits: List of qubit indices this gate acts on. Should contain exactly one index.
        """
        super().__init__(name="Y", qubits=qubits)
        if len(self.qubits) != 1:
            raise ValueError("Y gate acts on exactly one qubit.")
        # The matrix representation of the Y gate
        self.matrix = np.array([[0, -1j],
                                [1j, 0]], dtype=complex)

