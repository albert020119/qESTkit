# Initialize a quantum circuit with 2 qubits
from simulator.circuit.quantum_circuit import QuantumCircuit
from simulator.gates import Hadamard, CNOT

circuit = QuantumCircuit(num_qubits=2)

# Add some gates to the circuit
circuit.add_gate(Hadamard(qubits=[0]))  # Apply Hadamard on qubit 0
circuit.add_gate(CNOT(control_qubit=0, target_qubit=1))  # Apply CNOT

# Run the circuit and visualize the result
circuit.run_circuit()