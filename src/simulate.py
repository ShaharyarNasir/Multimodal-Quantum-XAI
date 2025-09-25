# src/simulate.py
import numpy as np
from qutip import basis, tensor, sigmaz, sigmax, mesolve, qeye


def simulate_quantum(params):
    try:
        num_qubits = params.get("num_qubits", 2)
        max_time = params.get("max_time", 10)
        num_points = params.get("num_points", 100)
        decoherence_rate = params.get("decoherence_rate", 0.0)

        if not 2 <= num_qubits <= 6:
            raise ValueError("num_qubits must be 2-6")
        if max_time <= 0 or num_points <= 0:
            raise ValueError("max_time and num_points must be positive")
        if decoherence_rate < 0:
            raise ValueError("decoherence_rate must be non-negative")

        H_terms = []
        for i in range(num_qubits - 1):
            ops = [qeye(2)] * num_qubits
            ops[i] = sigmaz()
            ops[i + 1] = sigmax()
            H_terms.append(tensor(ops))
            ops = [qeye(2)] * num_qubits
            ops[i] = sigmax()
            ops[i + 1] = sigmaz()
            H_terms.append(tensor(ops))

        H_field = sum(0.5 * tensor([sigmaz() if j == i else qeye(2)
                      for j in range(num_qubits)]) for i in range(num_qubits))
        H = sum(H_terms) + H_field

        states = [(basis(2, 0) + basis(2, 1)).unit()] + [basis(2, 0)
                                                         for _ in range(num_qubits - 1)]
        psi0 = tensor(states)

        ops_obs = [qeye(2)] * num_qubits
        ops_obs[0] = sigmax()
        ops_obs[-1] = sigmax()
        obs = tensor(ops_obs)

        times = np.linspace(0, max_time, num_points)

        c_ops = []
        if decoherence_rate > 0:
            for i in range(num_qubits):
                ops = [qeye(2)] * num_qubits
                ops[i] = sigmax()
                c_ops.append(np.sqrt(decoherence_rate) * tensor(ops))

        result = mesolve(H, psi0, times, c_ops, [obs])
        quantum_data = np.array(result.expect[0])
        return times, quantum_data
    except Exception as e:
        raise RuntimeError(f"Simulation failed: {str(e)}")
