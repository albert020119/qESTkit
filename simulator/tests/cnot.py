import numpy as np

from simulator.gates import CNOT

def test_cnot_gate():
    # Test CNOT gate with |00⟩ state
    state_vector = np.array([1, 0, 0, 0], dtype=complex)
    cnot_gate = CNOT(0,1)
    new_state_vector = cnot_gate.apply(state_vector)
    expected_state_vector = np.array([1, 0, 0, 0], dtype=complex)
    assert np.allclose(new_state_vector, expected_state_vector), "CNOT failed for |00⟩ state."

    # Test CNOT gate with |01⟩ state
    state_vector = np.array([0, 1, 0, 0], dtype=complex)
    new_state_vector = cnot_gate.apply(state_vector)
    expected_state_vector = np.array([0, 1, 0, 0], dtype=complex)
    assert np.allclose(new_state_vector, expected_state_vector), "CNOT failed for |01⟩ state."

    # Test CNOT gate with |10⟩ state
    state_vector = np.array([0, 0, 1, 0], dtype=complex)
    new_state_vector = cnot_gate.apply(state_vector)
    expected_state_vector = np.array([0, 0, 0, 1], dtype=complex)
    assert np.allclose(new_state_vector, expected_state_vector), "CNOT failed for |10⟩ state."

    # Test CNOT gate with |11⟩ state
    state_vector = np.array([0, 0, 0, 1], dtype=complex)
    new_state_vector = cnot_gate.apply(state_vector)
    expected_state_vector = np.array([0, 0, 1, 0], dtype=complex)
    assert np.allclose(new_state_vector, expected_state_vector), "CNOT failed for |11⟩ state."


if __name__ == "__main__":
    test_cnot_gate()
