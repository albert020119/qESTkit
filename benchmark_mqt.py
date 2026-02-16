import time
import matplotlib.pyplot as plt
from simulator.circuit.quantum_circuit import QuantumCircuit
from simulator.gates.hadamard import Hadamard
from simulator.gates.cnot import CNOT

def scalability_benchmark():
    qubit_counts = range(2, 11)  # Number of qubits from 2 to 10
    simulation_times = []

    for n in qubit_counts:
        # Create a quantum circuit with n qubits
        qc = QuantumCircuit(num_qubits=n)

        # Build a GHZ state: Hadamard on qubit 0, then cascading CNOTs
        qc.add_gate(Hadamard(qubits=[0]))
        for i in range(n - 1):
            qc.add_gate(CNOT(control_qubit=i, target_qubit=i + 1))

        # Measure the time taken to apply the circuit
        start_time = time.time()
        qc.apply()
        end_time = time.time()

        # Record the simulation time
        simulation_times.append(end_time - start_time)

    # Plot the results
    plt.plot(qubit_counts, simulation_times, marker='o')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Simulation Time (Seconds)')
    plt.title('MQT-Style Scalability Benchmark')
    plt.grid(True)
    plt.savefig('mqt_benchmark_result.png')
    plt.show()

if __name__ == '__main__':
    scalability_benchmark()