"""
Tests for the Qrisp → qESTkit translation layer.

Run with:  python -m pytest simulator/tests/qrisp_integration/ -v
  or:      python simulator/tests/qrisp_integration/test_translator.py
"""

import sys
import numpy as np

# ---------------------------------------------------------------------------
# Ensure the package root is on the path when running as a script
# ---------------------------------------------------------------------------
import pathlib
_root = str(pathlib.Path(__file__).resolve().parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

import qrisp

from simulator.qrisp_integration.translator import QrispTranslator
from simulator.runner import run_simulation


def _counts_to_probs(counts: dict, num_qubits: int) -> dict:
    """Normalise raw counts dict into probability dict keyed by bitstring."""
    total = sum(counts.values())
    probs = {}
    for k, v in counts.items():
        if isinstance(k, str):
            bitstr = k.replace("|", "").replace(">", "")
        else:
            bitstr = format(k, f"0{num_qubits}b")
        probs[bitstr] = v / total
    return probs


# -------------------------------------------------------------------
# 1. Bell state:  H on q0, then CX(q0, q1)  →  |00⟩ and |11⟩ ≈ 50/50
# -------------------------------------------------------------------
def test_bell_state():
    qv = qrisp.QuantumVariable(2)
    qrisp.h(qv[0])
    qrisp.cx(qv[0], qv[1])

    translator = QrispTranslator(transpile=True)
    qc = translator.from_qrisp_variable(qv)
    counts = run_simulation(qc, num_simulations=2000)
    probs = _counts_to_probs(counts, qc.num_qubits)

    assert "00" in probs and "11" in probs, f"Expected 00 and 11, got {probs}"
    assert probs["00"] > 0.35, f"|00⟩ prob too low: {probs['00']}"
    assert probs["11"] > 0.35, f"|11⟩ prob too low: {probs['11']}"
    # Should NOT have 01 or 10 (or negligible)
    assert probs.get("01", 0) < 0.05
    assert probs.get("10", 0) < 0.05
    print("  ✓ Bell state test passed")


# -------------------------------------------------------------------
# 2. Single X gate flips |0⟩ → |1⟩
# -------------------------------------------------------------------
def test_x_gate():
    qv = qrisp.QuantumVariable(1)
    qrisp.x(qv[0])

    translator = QrispTranslator()
    qc = translator.from_qrisp_variable(qv)
    counts = run_simulation(qc, num_simulations=100)
    probs = _counts_to_probs(counts, qc.num_qubits)

    assert probs.get("1", 0) > 0.99, f"Expected all |1⟩, got {probs}"
    print("  ✓ X gate test passed")


# -------------------------------------------------------------------
# 3. Hadamard → equal superposition
# -------------------------------------------------------------------
def test_hadamard():
    qv = qrisp.QuantumVariable(1)
    qrisp.h(qv[0])

    translator = QrispTranslator()
    qc = translator.from_qrisp_variable(qv)
    counts = run_simulation(qc, num_simulations=2000)
    probs = _counts_to_probs(counts, qc.num_qubits)

    assert probs.get("0", 0) > 0.35
    assert probs.get("1", 0) > 0.35
    print("  ✓ Hadamard test passed")


# -------------------------------------------------------------------
# 4. Parametric gate: Rx(π) ≈ X gate
# -------------------------------------------------------------------
def test_rx_pi():
    qv = qrisp.QuantumVariable(1)
    qrisp.rx(np.pi, qv[0])

    translator = QrispTranslator()
    qc = translator.from_qrisp_variable(qv)
    counts = run_simulation(qc, num_simulations=100)
    probs = _counts_to_probs(counts, qc.num_qubits)

    assert probs.get("1", 0) > 0.95, f"Rx(π) should flip to |1⟩, got {probs}"
    print("  ✓ Rx(π) test passed")


# -------------------------------------------------------------------
# 5. Supported-gates introspection
# -------------------------------------------------------------------
def test_supported_gates():
    gates = QrispTranslator.supported_gates()
    assert "h" in gates
    assert "cx" in gates
    assert "rz" in gates
    print(f"  ✓ Supported gates ({len(gates)}): {gates}")


# -------------------------------------------------------------------
# 6. Multi-qubit: 3-qubit GHZ state
# -------------------------------------------------------------------
def test_ghz_state():
    qv = qrisp.QuantumVariable(3)
    qrisp.h(qv[0])
    qrisp.cx(qv[0], qv[1])
    qrisp.cx(qv[1], qv[2])

    translator = QrispTranslator(transpile=True)
    qc = translator.from_qrisp_variable(qv)
    counts = run_simulation(qc, num_simulations=2000)
    probs = _counts_to_probs(counts, qc.num_qubits)

    assert probs.get("000", 0) > 0.35
    assert probs.get("111", 0) > 0.35
    # No other states should appear with significant probability
    for k, v in probs.items():
        if k not in ("000", "111"):
            assert v < 0.05, f"Unexpected state {k} with prob {v}"
    print("  ✓ GHZ state test passed")


if __name__ == "__main__":
    print("Running Qrisp integration tests...\n")
    test_bell_state()
    test_x_gate()
    test_hadamard()
    test_rx_pi()
    test_supported_gates()
    test_ghz_state()
    print("\nAll tests passed ✓")
