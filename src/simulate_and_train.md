# src/simulate_and_train.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from qutip import basis, tensor, sigmaz, sigmax, mesolve, qeye
import shap
import matplotlib.pyplot as plt
import io
import base64  


np.random.seed(42)
torch.manual_seed(42)


def simulate_and_train(num_qubits=2, max_time=10, num_points=100, epochs=100, decoherence_rate=0.0):
    """
    Simulate entangled qubits with optional decoherence,
    train a small neural network proxy, and compute SHAP values.
    Returns a 2x2 dashboard figure.
    """

    # Hamiltonian for n qubits (linear chain) with magnetic field
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

    # Fixed Magnetic field term: 0.5 * σ_z on each qubit
    H_field = sum(0.5 * tensor([sigmaz() if j == i else qeye(2) for j in range(num_qubits)]) for i in range(num_qubits))
    H = sum(H_terms) + H_field

    # Initial state: superposition on first qubit, rest |0>
    states = [basis(2, 0) for _ in range(num_qubits)]
    superpos_first = (basis(2, 0) + basis(2, 1)).unit()
    states[0] = superpos_first
    psi0 = tensor(states)

    # Observable: σ_x on endpoints for correlation
    obs_ops = [qeye(2)] * num_qubits
    obs_ops[0] = sigmax()
    obs_ops[-1] = sigmax()
    obs = tensor(obs_ops)

    times = np.linspace(0, max_time, num_points)

    # Optional decoherence: bit-flip channel
    c_ops = []
    if decoherence_rate > 0:
        for i in range(num_qubits):
            ops = [qeye(2)] * num_qubits
            ops[i] = sigmax()
            c_ops.append(np.sqrt(decoherence_rate) * tensor(ops))

    result = mesolve(H, psi0, times, c_ops, [obs])
    quantum_data = np.array(result.expect[0])
    inputs = times.reshape(-1, 1)
    outputs = quantum_data.reshape(-1, 1)

    # Neural network (scaled with qubit count)
    hidden1 = max(20, 10 * num_qubits)
    hidden2 = max(10, 5 * num_qubits)

    class QuantumNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1, hidden1)
            self.fc2 = nn.Linear(hidden1, hidden2)
            self.fc3 = nn.Linear(hidden2, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    model = QuantumNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32)

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(inputs_tensor)
        loss = loss_fn(preds, outputs_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # SHAP explanation on subset
    subset_size = min(20, num_points)
    explainer = shap.DeepExplainer(model, inputs_tensor[:subset_size])
    shap_values = explainer.shap_values(inputs_tensor[:subset_size], check_additivity=False)

    shap_fig = plt.figure()
    shap.summary_plot(shap_values, inputs[:subset_size], feature_names=["Time"], show=False)
    shap_buf = io.BytesIO()
    shap_fig.savefig(shap_buf, format='png')
    shap_base64 = base64.b64encode(shap_buf.getvalue()).decode('utf-8')
    plt.close(shap_fig)

    # Predictions
    with torch.no_grad():
        predicted = model(inputs_tensor).numpy()

    # Visualization: 2x2 dashboard
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Loss curve
    axs[0, 0].plot(range(epochs), losses)
    axs[0, 0].set_title("Training Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("MSE")

    # Prediction vs true
    axs[0, 1].plot(times, quantum_data, label="True Correlation")
    axs[0, 1].plot(times, predicted, '--', label="NN Prediction")
    axs[0, 1].set_title("Quantum Correlation vs. NN Prediction")
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Correlation")
    axs[0, 1].legend()

    # SHAP plot (load and display as image)
    shap_img = io.BytesIO(base64.b64decode(shap_base64))
    axs[1, 0].imshow(plt.imread(shap_img))
    axs[1, 0].set_title("SHAP: Time Impact on Predictions")
    axs[1, 0].axis('off')

    # Summary text
    final_mse = loss_fn(preds, outputs_tensor).item()
    axs[1, 1].text(0.5, 0.5, f"Final MSE: {final_mse:.4f}\nQubits: {num_qubits}\nDecoherence: {decoherence_rate}",
                   ha='center', va='center', transform=axs[1, 1].transAxes)
    axs[1, 1].set_title("Summary")
    axs[1, 1].axis('off')

    plt.tight_layout()
    return fig