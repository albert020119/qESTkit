from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List, Literal, Dict

from starlette.middleware.cors import CORSMiddleware

from simulator.gates import Hadamard, X, CZ, CNOT, Identity, Ph, Rx, Ry, Rz, S, T, X, Y, Z
from simulator.circuit.quantum_circuit import QuantumCircuit
from simulator.runner import run_simulation, run_noisy_simulation

app = FastAPI(title="Quantum Simulator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define allowed gate names
GATE_NAME = Literal[
    "H", "X", "Y", "Z", "S", "T",
    "CZ", "CNOT", "Identity", "Ph",
    "Rx", "Ry", "Rz"
]


# Input format for simulation requests
class GateInstruction(BaseModel):
    name: GATE_NAME
    qubits: List[int]


class SimulationRequest(BaseModel):
    num_qubits: int
    gates: List[GateInstruction]
    num_simulations: int = 100


class NoisySimulationRequest(SimulationRequest):
    gate_error_prob: float = 0.01
    measurement_error_prob: float = 0.02


def build_circuit(request: SimulationRequest) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits=request.num_qubits)
    for gate in request.gates:
        if gate.name == "H":
            qc.add_gate(Hadamard(qubits=gate.qubits))
        elif gate.name == "X":
            qc.add_gate(X(qubits=gate.qubits))
        elif gate.name == "Y":
            qc.add_gate(Y(qubits=gate.qubits))
        elif gate.name == "Z":
            qc.add_gate(Z(qubits=gate.qubits))
        elif gate.name == "S":
            qc.add_gate(S(qubits=gate.qubits))
        elif gate.name == "T":
            qc.add_gate(T(qubits=gate.qubits))
        elif gate.name == "CZ":
            qc.add_gate(CZ(qubits=gate.qubits))
        elif gate.name == "CNOT":
            qc.add_gate(CNOT(control_qubit=gate.qubits[0], target_qubit=gate.qubits[1]))
        elif gate.name == "Identity":
            qc.add_gate(Identity(qubits=gate.qubits))
        elif gate.name == "Ph":
            qc.add_gate(Ph(qubits=gate.qubits))
        elif gate.name == "Rx":
            qc.add_gate(Rx(qubits=gate.qubits))
        elif gate.name == "Ry":
            qc.add_gate(Ry(qubits=gate.qubits))
        elif gate.name == "Rz":
            qc.add_gate(Rz(qubits=gate.qubits))
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported gate: {gate.name}")
    return qc



@app.post("/simulate")
def simulate(request: SimulationRequest) -> Dict[str, int]:
    qc = build_circuit(request)
    return run_simulation(qc, num_simulations=request.num_simulations)


@app.post("/simulate-noisy")
def simulate_noisy(request: NoisySimulationRequest) -> Dict[str, int]:
    qc = build_circuit(request)
    return run_noisy_simulation(
        qc,
        num_simulations=request.num_simulations,
        gate_error_prob=request.gate_error_prob,
        measurement_error_prob=request.measurement_error_prob
    )


@app.get("/")
def root():
    return {"message": "Quantum Simulator API is up. Use /simulate or /simulate-noisy."}


if __name__ == "__main__":
    uvicorn.run(app)