# src/nn_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def train_nn(times, quantum_data, epochs=1000):
    np.random.seed(42)
    torch.manual_seed(42)
    num_qubits = len(quantum_data.shape) if len(quantum_data.shape) > 1 else 2
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
    # Normalize inputs/outputs for SHAP stability
    inputs = times.reshape(-1, 1) / times.max()  # Normalize time to [0,1]
    outputs = quantum_data.reshape(-1, 1) / max(abs(quantum_data.max()),
                                                # Normalize to [-1,1]
                                                abs(quantum_data.min()))
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

    with torch.no_grad():
        predicted = model(inputs_tensor).numpy().flatten() * \
            max(abs(quantum_data.max()), abs(quantum_data.min()))
    final_mse = loss_fn(torch.tensor(predicted).reshape(-1, 1),
                        torch.tensor(quantum_data).reshape(-1, 1)).item()
    return model, predicted, losses, final_mse
