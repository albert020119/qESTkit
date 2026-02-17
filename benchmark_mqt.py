import time
import matplotlib.pyplot as plt
from simulator.circuit.quantum_circuit import QuantumCircuit
from simulator.gates.hadamard import Hadamard
from simulator.gates.cnot import CNOT
from simulator.runner import run_simulation, run_noisy_simulation
import numpy as np

def hellinger_fidelity(ideal_counts, noisy_counts):
    """
    Calculate the Hellinger fidelity between two distributions.

    ret  Hellinger fidelity value (0.0 to 1.0).
    """
    # Normalize the counts to probabilities
    ideal_total = sum(ideal_counts.values())
    noisy_total = sum(noisy_counts.values())

    ideal_probs = {k: v / ideal_total for k, v in ideal_counts.items()}
    noisy_probs = {k: v / noisy_total for k, v in noisy_counts.items()}

    # Calculate the Hellinger fidelity
    fidelity = sum(
        np.sqrt(ideal_probs.get(k, 0) * noisy_probs.get(k, 0))
        for k in set(ideal_probs.keys()).union(noisy_probs.keys())
    ) ** 2

    return fidelity

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
        run_simulation(qc, num_simulations=1000)
        end_time = time.time()

        # Record the simulation time
        simulation_times.append(end_time - start_time)

    # Plot Simulation Time
    plt.figure(figsize=(10, 5))
    plt.plot(qubit_counts, simulation_times, marker='o', label='Simulation Time')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Simulation Time (Seconds)')
    plt.title('MQT-Style Scalability Analysis')
    plt.grid(True)
    plt.legend()
    plt.savefig('mqt_benchmark_result.png')
    plt.close()

def quality_benchmark():
    qubit_counts = range(2, 11)  # Number of qubits from 2 to 10
    fidelities = []

    for n in qubit_counts:
        # Create a quantum circuit with n qubits
        qc = QuantumCircuit(num_qubits=n)

        # Build a GHZ state: Hadamard on qubit 0, then cascading CNOTs
        qc.add_gate(Hadamard(qubits=[0]))
        for i in range(n - 1):
            qc.add_gate(CNOT(control_qubit=i, target_qubit=i + 1))

        # Run the ideal simulation
        ideal_counts = run_simulation(qc, num_simulations=1000)

        # Run the noisy simulation with the ibm_brisbane profile
        noisy_counts = run_noisy_simulation(
            qc, num_simulations=1000, 
            gate_error_prob=0.012, measurement_error_prob=0.02
        )

        # Calculate the Hellinger fidelity
        fidelity = hellinger_fidelity(ideal_counts, noisy_counts)
        fidelities.append(fidelity)

    # Plot Hellinger Fidelity
    plt.figure(figsize=(10, 5))
    plt.plot(qubit_counts, fidelities, marker='o', color='r', label='Hellinger Fidelity')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Hellinger Fidelity')
    plt.ylim(0.0, 1.0)
    plt.title('MQT-Style Quality Analysis')
    plt.grid(True)
    plt.legend()
    plt.savefig('mqt_advanced_benchmark.png')
    plt.close()

if __name__ == '__main__':
    scalability_benchmark()
    quality_benchmark()