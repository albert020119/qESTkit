import numpy as np

from simulator.circuit.quantum_circuit import QuantumCircuit
from simulator.gates import Hadamard, T


def test_double_hadamard():
    # Create a single-qubit circuit
    qc = QuantumCircuit(num_qubits=2)

    # Add two Hadamard gates to the circuit
    qc.add_gate(Hadamard(qubits=[0]))
    qc.add_gate(Hadamard(qubits=[1]))
    qc.add_gate(Hadamard(qubits=[1]))


    # Apply the circuit
    for x in range(1000):
        qc.apply()
        print(qc.measure_state())
        qc.reset()
        qc.add_gate(Hadamard(qubits=[0]))
        qc.add_gate(Hadamard(qubits=[1]))
        qc.add_gate(Hadamard(qubits=[1]))

    print("State vector after applying gates:", qc.state)
    # Expected state: Apply Hadamard to |0> and |1> sequentially
    expected_state = np.array([1/np.sqrt(2), 0, 1/np.sqrt(2), 0], dtype=complex)

    # Verify the final state
    assert np.allclose(qc.state, expected_state), "Double Hadamard test failed!"

def test_hadamard_t():
    # Create a single-qubit circuit
    qc = QuantumCircuit(num_qubits=1)

    # Add a Hadamard gate followed by a T gate
    qc.add_gate(Hadamard(qubits=[0]))
    qc.add_gate(T(qubits=[0]))

    # Apply the circuit
    qc.apply()

    # Expected state: Apply T and H sequentially
    h_gate = Hadamard(qubits=[0]).matrix
    t_gate = T(qubits=[0]).matrix
    intermediate_state = h_gate @ np.array([1, 0], dtype=complex)
    expected_state = t_gate @ intermediate_state

    # Verify the final state
    assert np.allclose(qc.state, expected_state), "Hadamard + T gate test failed!"

if __name__ == "__main__":
    test_double_hadamard()
    test_hadamard_t()
