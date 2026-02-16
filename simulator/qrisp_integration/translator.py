"""
Translator that converts a Qrisp-compiled QuantumCircuit into the
qESTkit simulator's internal QuantumCircuit + Gate objects.

Qrisp compiles high-level programs into its own QuantumCircuit whose
``.data`` list contains ``Instruction`` objects.  Each instruction carries:
  - ``op.name``   – lowercase gate name  (``"h"``, ``"cx"``, ``"rz"``, …)
  - ``op.params`` – list of float parameters (empty for non-parametric gates)
  - ``qubits``    – list of ``Qubit`` objects

The translator:
  1. Calls ``transpile()`` on the Qrisp circuit so exotic / composite gates
     are decomposed into a basis the simulator already supports.
  2. Builds a qubit-index map  (Qrisp ``Qubit`` → ``int``).
  3. Walks every instruction and instantiates the matching qESTkit ``Gate``.
  4. Returns a ready-to-simulate ``QuantumCircuit``.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, TYPE_CHECKING

from simulator.circuit.quantum_circuit import QuantumCircuit
from simulator.gates import (
    Hadamard, X, Y, Z, S, T,
    CNOT, CZ,
    Rx, Ry, Rz,
    Ph, Identity,
)

if TYPE_CHECKING:  # avoid hard import at module level for flexibility
    import qrisp as _qrisp

# ---------------------------------------------------------------------------
# Gate-name mapping: Qrisp op.name  →  factory that returns a qESTkit Gate
# ---------------------------------------------------------------------------

def _make_h(qubits, _params):
    return Hadamard(qubits=qubits)

def _make_x(qubits, _params):
    return X(qubits=qubits)

def _make_y(qubits, _params):
    return Y(qubits=qubits)

def _make_z(qubits, _params):
    return Z(qubits=qubits)

def _make_s(qubits, _params):
    return S(qubits=qubits)

def _make_s_dg(qubits, _params):
    """S† = Phase(-π/2) gate."""
    return Ph(delta=-np.pi / 2, qubits=qubits)

def _make_t(qubits, _params):
    return T(qubits=qubits)

def _make_t_dg(qubits, _params):
    """T† = Phase(-π/4) gate."""
    return Ph(delta=-np.pi / 4, qubits=qubits)

def _make_id(qubits, _params):
    return Identity(qubits=qubits)

def _make_cx(qubits, _params):
    return CNOT(control_qubit=qubits[0], target_qubit=qubits[1])

def _make_cz(qubits, _params):
    return CZ(qubits=qubits)

def _make_rx(qubits, params):
    return Rx(theta=float(params[0]), qubits=qubits)

def _make_ry(qubits, params):
    return Ry(theta=float(params[0]), qubits=qubits)

def _make_rz(qubits, params):
    return Rz(theta=float(params[0]), qubits=qubits)

def _make_p(qubits, params):
    """Phase gate  p(θ) = diag(1, e^{iθ})."""
    return Ph(delta=float(params[0]), qubits=qubits)

def _make_sx(qubits, _params):
    """√X = Rx(π/2) up to global phase."""
    return Rx(theta=np.pi / 2, qubits=qubits)

def _make_sx_dg(qubits, _params):
    """(√X)† = Rx(-π/2) up to global phase."""
    return Rx(theta=-np.pi / 2, qubits=qubits)


# Central registry: Qrisp op name  →  factory callable
GATE_MAP = {
    "h":     _make_h,
    "x":     _make_x,
    "y":     _make_y,
    "z":     _make_z,
    "s":     _make_s,
    "s_dg":  _make_s_dg,
    "t":     _make_t,
    "t_dg":  _make_t_dg,
    "id":    _make_id,
    "cx":    _make_cx,
    "cz":    _make_cz,
    "rx":    _make_rx,
    "ry":    _make_ry,
    "rz":    _make_rz,
    "p":     _make_p,
    "sx":    _make_sx,
    "sx_dg": _make_sx_dg,
}

# Instructions produced by Qrisp that we silently skip (bookkeeping only)
_SKIP_OPS = {"qb_alloc", "qb_dealloc", "barrier", "measure"}


class QrispTranslator:
    """Translate Qrisp quantum programs into qESTkit's internal representation."""

    def __init__(self, transpile: bool = True):
        """
        Parameters
        ----------
        transpile : bool
            If ``True`` (default), call ``qrisp_circuit.transpile()`` before
            translation so that composite / exotic gates are decomposed into
            the basis set supported by the simulator.
        """
        self.transpile = transpile

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def from_qrisp_variable(self, qv: "_qrisp.QuantumVariable",
                            transpile: Optional[bool] = None) -> QuantumCircuit:
        """Compile & translate the circuit behind a ``QuantumVariable``.

        Parameters
        ----------
        qv : qrisp.QuantumVariable
            The high-level Qrisp variable whose quantum session will be compiled.
        transpile : bool | None
            Override the instance-level transpile flag for this call.

        Returns
        -------
        QuantumCircuit
            A qESTkit circuit ready for simulation.
        """
        qrisp_qc = qv.qs.compile()
        return self.translate(qrisp_qc, transpile=transpile)

    def from_qrisp_session(self, qs: "_qrisp.QuantumSession",
                           transpile: Optional[bool] = None) -> QuantumCircuit:
        """Compile & translate a full ``QuantumSession``."""
        qrisp_qc = qs.compile()
        return self.translate(qrisp_qc, transpile=transpile)

    # ------------------------------------------------------------------
    # Core translation
    # ------------------------------------------------------------------

    def translate(self, qrisp_circuit, transpile: Optional[bool] = None) -> QuantumCircuit:
        """Convert a *compiled* Qrisp ``QuantumCircuit`` into a qESTkit one.

        Parameters
        ----------
        qrisp_circuit : qrisp.circuit.quantum_circuit.QuantumCircuit
            The Qrisp circuit (output of ``qs.compile()``).
        transpile : bool | None
            Override the instance-level flag.

        Returns
        -------
        QuantumCircuit
        """
        do_transpile = transpile if transpile is not None else self.transpile
        if do_transpile:
            qrisp_circuit = qrisp_circuit.transpile()

        # Build qubit-index map
        qubit_map: Dict[object, int] = {
            qubit: idx for idx, qubit in enumerate(qrisp_circuit.qubits)
        }
        num_qubits = len(qubit_map)
        qc = QuantumCircuit(num_qubits)

        for instr in qrisp_circuit.data:
            name = instr.op.name.lower()

            if name in _SKIP_OPS:
                continue

            qubit_indices: List[int] = [qubit_map[q] for q in instr.qubits]
            params = list(instr.op.params)

            factory = GATE_MAP.get(name)
            if factory is None:
                raise ValueError(
                    f"Unsupported Qrisp gate '{instr.op.name}'. "
                    f"Consider enabling transpile=True to decompose it, "
                    f"or extend GATE_MAP in translator.py."
                )

            gate = factory(qubit_indices, params)
            qc.add_gate(gate)

        return qc

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @staticmethod
    def supported_gates() -> List[str]:
        """Return the Qrisp gate names the translator can handle."""
        return sorted(GATE_MAP.keys())
