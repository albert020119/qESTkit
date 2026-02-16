from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List, Literal, Dict
import os
import requests
from dotenv import load_dotenv
from google.cloud import firestore
import numpy as np

# Configure your key

from starlette.middleware.cors import CORSMiddleware

from simulator.gates import Hadamard, X, CZ, CNOT, Identity, Ph, Rx, Ry, Rz, S, T, Y, Z
from simulator.circuit.quantum_circuit import QuantumCircuit
from simulator.runner import run_simulation, run_noisy_simulation
from simulator.qrisp_integration import QrispTranslator, QESTkitBackend

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
    "CZ", "CNOT", "Identity", "I", "Ph", "PH",
    "Rx", "Ry", "Rz",
    "RX", "RY", "RZ",
    "SWAP"
]

# Canonical name mapping (handles case variants)
_GATE_ALIAS = {
    "H": "H", "X": "X", "Y": "Y", "Z": "Z",
    "S": "S", "T": "T",
    "CZ": "CZ", "CNOT": "CNOT",
    "IDENTITY": "Identity", "Identity": "Identity", "I": "Identity",
    "PH": "Ph", "Ph": "Ph",
    "RX": "Rx", "Rx": "Rx",
    "RY": "Ry", "Ry": "Ry",
    "RZ": "Rz", "Rz": "Rz",
    "SWAP": "SWAP",
}


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


class QrispCodeRequest(BaseModel):
    code: str
    shots: int = 1000


class UserMessage(BaseModel):
    message: str


class Report(BaseModel):
    code: str
    message: str


def build_circuit(request: SimulationRequest) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits=request.num_qubits)
    for gate in request.gates:
        # Normalise gate name so uppercase variants are accepted
        canonical = _GATE_ALIAS.get(gate.name, gate.name)
        if canonical == "H":
            qc.add_gate(Hadamard(qubits=gate.qubits))
        elif canonical == "X":
            qc.add_gate(X(qubits=gate.qubits))
        elif canonical == "Y":
            qc.add_gate(Y(qubits=gate.qubits))
        elif canonical == "Z":
            qc.add_gate(Z(qubits=gate.qubits))
        elif canonical == "S":
            qc.add_gate(S(qubits=gate.qubits))
        elif canonical == "T":
            qc.add_gate(T(qubits=gate.qubits))
        elif canonical == "CZ":
            qc.add_gate(CZ(qubits=gate.qubits))
        elif canonical == "CNOT":
            qc.add_gate(CNOT(control_qubit=gate.qubits[0], target_qubit=gate.qubits[1]))
        elif canonical == "Identity":
            qc.add_gate(Identity(qubits=gate.qubits))
        elif canonical == "Ph":
            qc.add_gate(Ph(qubits=gate.qubits))
        elif canonical == "Rx":
            theta = float(gate.param) * np.pi / 180 if gate.param else 0  # Convert degrees to radians
            qc.add_gate(Rx(theta=theta, qubits=gate.qubits))
        elif canonical == "Ry":
            theta = float(gate.param) * np.pi / 180 if gate.param else 0  # Convert degrees to radians
            qc.add_gate(Ry(theta=theta, qubits=gate.qubits))
        elif canonical == "Rz":
            theta = float(gate.param) * np.pi / 180 if gate.param else 0  # Convert degrees to radians
            qc.add_gate(Rz(theta=theta, qubits=gate.qubits))
        elif canonical == "SWAP":
            # Decompose SWAP into 3 CNOTs
            q0, q1 = gate.qubits[0], gate.qubits[1]
            qc.add_gate(CNOT(control_qubit=q0, target_qubit=q1))
            qc.add_gate(CNOT(control_qubit=q1, target_qubit=q0))
            qc.add_gate(CNOT(control_qubit=q0, target_qubit=q1))
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


@app.post("/simulate-qrisp")
def simulate_qrisp(request: QrispCodeRequest) -> Dict[str, int]:
    """
    Execute a Qrisp program on the qESTkit simulator.

    The ``code`` field must be valid Python that uses Qrisp to build a
    quantum program.  The code **must** assign the final
    ``qrisp.QuantumVariable`` (or a list of them) to a variable called
    ``result``.

    Example payload::

        {
            "code": "import qrisp\nqv = qrisp.QuantumVariable(2)\nqrisp.h(qv[0])\nqrisp.cx(qv[0], qv[1])\nresult = qv",
            "shots": 1000
        }
    """
    try:
        import qrisp  # noqa: F401 – make qrisp available inside exec'd code

        # Provide a controlled namespace for the user code
        exec_globals: Dict = {"qrisp": qrisp, "__builtins__": __builtins__}
        exec(request.code, exec_globals)

        result_var = exec_globals.get("result")
        if result_var is None:
            raise HTTPException(
                status_code=400,
                detail="Qrisp code must assign the output QuantumVariable to 'result'.",
            )

        translator = QrispTranslator(transpile=True)

        # Support both a single QuantumVariable and a list of them
        if isinstance(result_var, qrisp.QuantumVariable):
            qc = translator.from_qrisp_variable(result_var)
        else:
            raise HTTPException(
                status_code=400,
                detail="'result' must be a qrisp.QuantumVariable.",
            )

        counts = run_simulation(qc, num_simulations=request.shots)
        # Normalise keys to plain bitstrings
        clean: Dict[str, int] = {}
        for key, count in counts.items():
            if isinstance(key, str):
                bitstr = key.replace("|", "").replace(">", "")
            else:
                bitstr = format(key, f"0{qc.num_qubits}b")
            clean[bitstr] = count
        return clean

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Qrisp execution error: {exc}")


@app.post("/ask-gemini")
def ask_gemini(user_message: UserMessage):
    api_key = gemini_api_key
    if not api_key:
        raise HTTPException(status_code=500, detail="Gemini API key not set.")

    # Use a standard model name confirmed to work with system instructions
    model_name = "gemini-2.5-flash"

    system_prompt = (
        "You are a helpful assistant for a Quantum Simulator application. "
        "Your goal is to help users understand quantum computing concepts, quantum gates, and circuits. "
        "You should also be able to explain code snippets provided by the user. "
        "Keep your answers concise (2-3 sentences) unless a detailed code explanation is required. "
        "If the user asks about topics unrelated to quantum computing or programming, politely decline."
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    data = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_message.message}]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 300
        }
    }

    try:
        # Using requests inside an async function blocks the event loop, 
        # but for simplicity we keep it unless you want to switch to httpx.
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        gemini_response = response.json()

        try:
            answer = gemini_response["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            answer = "I'm sorry, I couldn't generate a response."

        return {"answer": answer}

    except requests.exceptions.HTTPError as e:
        print(f"Gemini API Error: {e}")
        if e.response is not None:
            print(f"Response content: {e.response.text}")
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


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