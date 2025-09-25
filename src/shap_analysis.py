import shap
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import torch


def generate_shap_and_plot(model, times, quantum_data, predicted, losses, final_mse, num_qubits, decoherence_rate):
    try:
        # Normalize times for SHAP
        times_normalized = times / times.max()
        subset_size = min(10, len(times))  # Smaller for stability
        background = torch.tensor(
            times_normalized[:subset_size].reshape(-1, 1), dtype=torch.float32)

        # Use KernelExplainer for robustness
        def model_predict(inputs):
            with torch.no_grad():
                return model(torch.tensor(inputs, dtype=torch.float32)).numpy()

        explainer = shap.KernelExplainer(
            model_predict, times_normalized[:subset_size].reshape(-1, 1))
        shap_values = explainer.shap_values(
            times_normalized[:subset_size].reshape(-1, 1), nsamples=100)

        # SHAP plot: Pass 2D features to fix 'tuple index out of range'
        shap_fig = plt.figure(figsize=(6, 4))
        shap.summary_plot(
            shap_values, times[:subset_size].reshape(-1, 1), feature_names=["Time"], show=False)
        shap_buf = io.BytesIO()
        shap_fig.savefig(shap_buf, format='png', bbox_inches='tight')
        shap_buf.seek(0)
        shap_base64 = base64.b64encode(shap_buf.getvalue()).decode('utf-8')
        plt.close(shap_fig)

        # Dashboard figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].plot(losses)
        axs[0, 0].set_title("Training Loss")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].set_ylabel("MSE")

        axs[0, 1].plot(times, quantum_data, label="True")
        axs[0, 1].plot(times, predicted, '--', label="NN")
        axs[0, 1].legend()
        axs[0, 1].set_title("Prediction vs True")
        axs[0, 1].set_xlabel("Time")
        axs[0, 1].set_ylabel("Correlation")

        # Embed SHAP image: Seek to 0 before reading
        shap_img = io.BytesIO(base64.b64decode(shap_base64))
        shap_img.seek(0)  # Ensure position is at start
        axs[1, 0].imshow(plt.imread(shap_img))
        axs[1, 0].set_title("SHAP: Time Impact")
        axs[1, 0].axis('off')

        axs[1, 1].text(0.5, 0.5, f"MSE: {final_mse:.4f}\nQubits: {num_qubits}\nDeco: {decoherence_rate}",
                       ha='center', va='center')
        axs[1, 1].set_title("Summary")
        axs[1, 1].axis('off')

        plt.tight_layout()
        return fig

    except Exception as e:
        logger.error(f"SHAP failed: {str(e)}")
        # Fallback plot
        fig, ax = plt.subplots()
        ax.plot(times, quantum_data, label="True")
        ax.plot(times, predicted, label="NN")
        ax.set_title(f"Prediction (MSE: {final_mse:.4f})")
        ax.legend()
        return fig
