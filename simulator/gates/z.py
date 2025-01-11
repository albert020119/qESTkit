import numpy as np
from simulator.gates.gate import Gate


class Z(Gate):
    def __init__(self, qubits):
        """
        Initialize the Z gate.

        :param qubits: List of qubit indices this gate acts on. Should contain exactly one index.
        """
        super().__init__(name="Z", qubits=qubits)
        if len(self.qubits) != 1:
            raise ValueError("Z gate acts on exactly one qubit.")
        # The matrix representation of the Z gate
        self.matrix = np.array([[1, 0],
                                [0, -1]], dtype=complex)
