"""
Tests for the QESTkitBackend (Qrisp backend adapter).

Run with:  python simulator/tests/qrisp_integration/test_backend.py
"""

import sys
import pathlib

_root = str(pathlib.Path(__file__).resolve().parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

import qrisp
from simulator.qrisp_integration.backend import QESTkitBackend


def test_backend_bell():
    """Run a Bell circuit via the backend interface."""
    backend = QESTkitBackend()

    qv = qrisp.QuantumVariable(2)
    qrisp.h(qv[0])
    qrisp.cx(qv[0], qv[1])

    qc = qv.qs.compile()
    job = backend.run(qc, shots=2000)
    counts = job.result().get_counts()

    total = sum(counts.values())
    probs = {k: v / total for k, v in counts.items()}

    assert probs.get("00", 0) > 0.35
    assert probs.get("11", 0) > 0.35
    print(f"  ✓ Backend Bell test passed  counts={counts}")


def test_backend_x():
    """X gate through the backend."""
    backend = QESTkitBackend()

    qv = qrisp.QuantumVariable(1)
    qrisp.x(qv[0])

    qc = qv.qs.compile()
    job = backend.run(qc, shots=100)
    counts = job.result().get_counts()
    assert counts.get("1", 0) == 100
    print(f"  ✓ Backend X test passed  counts={counts}")


if __name__ == "__main__":
    print("Running QESTkitBackend tests...\n")
    test_backend_bell()
    test_backend_x()
    print("\nAll backend tests passed ✓")
