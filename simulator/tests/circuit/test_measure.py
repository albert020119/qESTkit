import numpy as np
from collections import Counter
from simulator.circuit.quantum_circuit import QuantumCircuit
from simulator.gates.hadamard import Hadamard
from simulator.gates.cnot import CNOT

def test_complex_circuit_with_measurement():
    num_qubits = 2

    qc = QuantumCircuit(num_qubits=num_qubits)


    qc.add_gate(Hadamard(qubits=[0]))
    qc.add_gate(CNOT(0,1))

    qc.apply()
    qc.draw()
    print("State vector after applying gates:", qc.state)

    num_measurements = 1000
    measurement_results = []

    for _ in range(num_measurements):
        collapsed_state, measured_basis_state = qc.measure_state()
        measurement_results.append(measured_basis_state)

        # Reset the circuit to its original state for the next measurement
        qc.reset()
        qc.add_gate(Hadamard(qubits=[0]))
        qc.add_gate(CNOT(0,1))
        qc.apply()

    counts = Counter(measurement_results)

    probabilities = {state: count / num_measurements for state, count in counts.items()}

    print("Measurement results (counts):", counts)
    print("Empirical probabilities:", probabilities)

    expected_probabilities = {0: 0.5, 3: 0.5}
    for state, expected_prob in expected_probabilities.items():
        assert np.isclose(probabilities.get(state, 0), expected_prob, atol=0.1), (
            f"Probability for state {state} deviates from expected value."
        )

    print("Test passed: Empirical probabilities match expected values.")


if __name__ == "__main__":
    test_complex_circuit_with_measurement()

