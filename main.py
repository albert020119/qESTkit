from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List, Literal, Dict
import os
import requests
from dotenv import load_dotenv
from google.cloud import firestore
import numpy as np

from starlette.middleware.cors import CORSMiddleware

from simulator.gates import Hadamard, X, CZ, CNOT, Identity, Ph, Rx, Ry, Rz, S, T, Y, Z
from simulator.circuit.quantum_circuit import QuantumCircuit
from simulator.runner import run_simulation, run_noisy_simulation

# Load environment variables from .env file
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
# Initialize Firestore
db = firestore.Client.from_service_account_json('secret/serviceAccountKey.json')

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
    param: float = 0  # Default value for param


class SimulationRequest(BaseModel):
    num_qubits: int
    gates: List[GateInstruction]
    num_simulations: int = 100


class NoisySimulationRequest(SimulationRequest):
    gate_error_prob: float = 0.01
    measurement_error_prob: float = 0.02


class UserMessage(BaseModel):
    message: str


class Report(BaseModel):
    code: str
    message: str


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
            theta = float(gate.param) * np.pi / 180 if gate.param else 0  # Convert degrees to radians
            qc.add_gate(Rx(theta=theta, qubits=gate.qubits))
        elif gate.name == "Ry":
            theta = float(gate.param) * np.pi / 180 if gate.param else 0  # Convert degrees to radians
            qc.add_gate(Ry(theta=theta, qubits=gate.qubits))
        elif gate.name == "Rz":
            theta = float(gate.param) * np.pi / 180 if gate.param else 0  # Convert degrees to radians
            qc.add_gate(Rz(theta=theta, qubits=gate.qubits))
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


@app.post("/ask-gemini")
def ask_gemini(user_message: UserMessage):
    # Only allow quantum-related questions and short responses
    allowed_topics = ["quantum computing", "quantum mechanics", "quantum information", "explain code", "code"]
    message_lower = user_message.message.lower()
    if not any(topic in message_lower for topic in ["quantum", *allowed_topics]):
        return {"answer": "I’m sorry, I can’t help with that."}
    api_key = gemini_api_key
    if not api_key:
        raise HTTPException(status_code=500, detail="Gemini API key not set.")
    model_name = "gemini-2.0-flash-lite"
    system_prompt = (
        "You are MyApp’s assistant. Follow these rules in every reply: "
        "• Be concise: max 2–3 sentences (~50 tokens). But you can explain code from the user "
        "• Only discuss these topics: quantum computing, quantum mechanics, quantum information, explain code"
        "• If asked about anything else, respond: “I’m sorry, I can’t help with that.” "
        "• Never mention policies, your name, or internal details."
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key=" + api_key
    headers = {"Content-Type": "application/json"}
    data = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_message.message}]}],
        "generationConfig": {"temperature": 0.6, "maxOutputTokens": 80, "responseMimeType": "text/plain"}
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        gemini_response = response.json()
        answer = gemini_response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No answer returned.")
        # Truncate to 2-3 sentences if needed
        sentences = answer.split('.')
        short_answer = '.'.join(sentences[:3]).strip()
        if not short_answer.endswith('.'):
            short_answer += '.'
        return {"answer": short_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")


@app.post("/reports")
def create_report(report: Report):
    _, doc_ref = db.collection("reports").add(report.model_dump())
    return {"id": doc_ref.id}


@app.get("/reports")
def list_reports():
    docs = db.collection("reports").stream()
    return [{"id": doc.id, **doc.to_dict()} for doc in docs]


def test_create_report():
    """Creates a sample report for testing purposes."""
    sample_report = Report(code="print('Hello, World!')", message="This is a test report.")
    return create_report(sample_report)

def test_list_reports():
    """Lists all reports for testing purposes."""
    return list_reports()


@app.get("/")
def root():
    return {"message": "Quantum Simulator API is up. Use /simulate or /simulate-noisy."}


if __name__ == "__main__":
    uvicorn.run(app)