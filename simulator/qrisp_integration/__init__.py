"""
Qrisp Integration Module for qESTkit Simulator.

Provides a translation layer that converts Qrisp-generated quantum programs
into the simulator's internal representation, enabling high-level quantum
algorithm development via Qrisp while executing on the custom simulator.

Usage:
    from simulator.qrisp_integration import QrispTranslator, QESTkitBackend

    # Option 1 - Translate a Qrisp QuantumVariable's compiled circuit
    translator = QrispTranslator()
    qestkit_circuit = translator.from_qrisp_variable(qv)
    results = run_simulation(qestkit_circuit)

    # Option 2 - Use the backend directly so Qrisp calls our simulator
    backend = QESTkitBackend()
    results = qv.get_measurement(backend=backend)
"""

from .translator import QrispTranslator
from .backend import QESTkitBackend

__all__ = ['QrispTranslator', 'QESTkitBackend']
