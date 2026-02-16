"""
Custom backend that lets Qrisp execute circuits on the qESTkit simulator.

Qrisp's ``QuantumVariable.get_measurement()`` accepts a ``backend`` keyword.
A backend must expose a ``run(qc, shots)`` method that returns a results
object with a ``get_counts()`` method.

This module provides ``QESTkitBackend`` which:
  1. Receives a Qrisp ``QuantumCircuit``.
  2. Uses ``QrispTranslator`` to convert it to a qESTkit ``QuantumCircuit``.
  3. Runs the simulation via ``run_simulation``.
  4. Returns results in the format Qrisp expects.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Optional

from simulator.runner import run_simulation
from .translator import QrispTranslator


class _QESTkitResult:
    """Thin wrapper that satisfies the interface Qrisp expects from
    ``backend.run(qc, shots).result().get_counts()``."""

    def __init__(self, counts: Dict[str, int]):
        self._counts = counts

    def get_counts(self) -> Dict[str, int]:
        return self._counts


class _QESTkitJob:
    """Mimics a job object whose ``.result()`` returns our result wrapper."""

    def __init__(self, counts: Dict[str, int]):
        self._result = _QESTkitResult(counts)

    def result(self) -> _QESTkitResult:
        return self._result


class QESTkitBackend:
    """Drop-in backend for Qrisp that uses the qESTkit simulator.

    Example
    -------
    >>> import qrisp
    >>> from simulator.qrisp_integration import QESTkitBackend
    >>> backend = QESTkitBackend()
    >>> qv = qrisp.QuantumVariable(2)
    >>> qrisp.h(qv[0])
    >>> qrisp.cx(qv[0], qv[1])
    >>> print(qv.get_measurement(backend=backend))
    {'00': 0.5, '11': 0.5}
    """

    def __init__(self, transpile: bool = True):
        self._translator = QrispTranslator(transpile=transpile)

    def run(self, qrisp_circuit, shots: int = 1000, **kwargs) -> _QESTkitJob:
        """Translate and simulate a Qrisp circuit.

        Parameters
        ----------
        qrisp_circuit : qrisp.circuit.quantum_circuit.QuantumCircuit
            Compiled Qrisp circuit.
        shots : int
            Number of measurement samples.

        Returns
        -------
        _QESTkitJob
            A job-like object whose ``result().get_counts()`` returns a dict
            mapping bitstrings to counts.
        """
        qestkit_qc = self._translator.translate(qrisp_circuit)
        raw_counts = run_simulation(qestkit_qc, num_simulations=shots)

        # run_simulation returns keys like "|010>" – convert to plain
        # bitstrings that Qrisp expects ("010").
        clean_counts: Dict[str, int] = {}
        for key, count in raw_counts.items():
            if isinstance(key, str):
                bitstr = key.replace("|", "").replace(">", "")
            else:
                # key might be an int index
                bitstr = format(key, f"0{qestkit_qc.num_qubits}b")
            clean_counts[bitstr] = count

        return _QESTkitJob(clean_counts)
