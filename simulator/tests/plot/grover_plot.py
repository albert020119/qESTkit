# Create a quantum circuit with 3 qubits (2 data qubits + 1 auxiliary qubit)
from simulator.circuit.quantum_circuit import QuantumCircuit
from simulator.gates import Hadamard, CZ, X, CNOT

# Create a quantum circuit with 3 qubits
qc = QuantumCircuit(num_qubits=3)

# Step 1: Create superposition
qc.add_gate(Hadamard(qubits=[0]))
qc.add_gate(Hadamard(qubits=[1]))
qc.add_gate(Hadamard(qubits=[2]))

# Step 2: Introduce entanglement
qc.add_gate(CNOT(control_qubit=0, target_qubit=1))  # Entangle qubit 0 and 1
qc.add_gate(CNOT(control_qubit=1, target_qubit=2))  # Entangle qubit 1 and 2

# Step 3: Apply a conditional phase shift
qc.add_gate(CZ(qubits=[0, 2]))  # Phase flip if both qubit 0 and qubit 2 are |1>
qc.add_gate(X(qubits=[2]))  # Flip qubit 2
qc.add_gate(CZ(qubits=[1, 2]))  # Phase flip based on qubit 1 and flipped qubit 2
qc.add_gate(X(qubits=[2]))  # Undo flip on qubit 2

# Step 4: Diffusion-like operation
qc.add_gate(Hadamard(qubits=[0]))
qc.add_gate(Hadamard(qubits=[1]))
qc.add_gate(Hadamard(qubits=[2]))
qc.add_gate(X(qubits=[0]))
qc.add_gate(X(qubits=[1]))
qc.add_gate(X(qubits=[2]))
qc.add_gate(CZ(qubits=[0, 2]))
qc.add_gate(X(qubits=[0]))
qc.add_gate(X(qubits=[1]))
qc.add_gate(X(qubits=[2]))
qc.add_gate(Hadamard(qubits=[0]))
qc.add_gate(Hadamard(qubits=[1]))
qc.add_gate(Hadamard(qubits=[2]))

# Run the circuit
qc.run_circuit()
