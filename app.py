# app.py
import gradio as gr
from src.simulate import simulate_quantum
from src.nn_model import train_nn
from src.shap_analysis import generate_shap_and_plot
import re


def parse_query(query):
    try:
        query = query.lower().strip()
        params = {"num_qubits": 2, "max_time": 10, "num_points": 100,
                  "epochs": 100, "decoherence_rate": 0.0}
        if not query:
            return params
        # Extract qubits
        qubit_match = re.search(r'(\d+)\s*(qubit|qubits)', query)
        if qubit_match:
            num_qubits = int(qubit_match.group(1))
            if 2 <= num_qubits <= 6:
                params["num_qubits"] = num_qubits
            else:
                raise ValueError("Number of qubits must be 2-6")
        # Extract time
        time_match = re.search(r'(\d+)\s*(time|unit|units)', query)
        if time_match:
            params["max_time"] = int(time_match.group(1))
        return params
    except Exception as e:
        return f"Parsing error: {str(e)}"


def quantum_xai_app(query, num_qubits, max_time, num_points, epochs, decoherence_rate):
    try:
        if query.strip():
            params = parse_query(query)
            if isinstance(params, str):
                return None, params
        else:
            params = {
                "num_qubits": num_qubits,
                "max_time": max_time,
                "num_points": num_points,
                "epochs": epochs,
                "decoherence_rate": decoherence_rate
            }
        # Debug print (remove later)
        print(f"Running simulation with params: {params}")
        times, quantum_data = simulate_quantum(params)
        model, predicted, losses, final_mse = train_nn(
            times, quantum_data, params["epochs"])
        fig = generate_shap_and_plot(model, times, quantum_data, predicted, losses, final_mse,
                                     params["num_qubits"], params["decoherence_rate"])
        explanation = f"Simulation for {params['num_qubits']} qubits, time 0-{params['max_time']}. MSE: {final_mse:.4f}"
        return fig, explanation
    except Exception as e:
        print(f"App error: {str(e)}")  # Debug print
        return None, f"Error: {str(e)}"


with gr.Blocks(title="Multimodal Quantum XAI App v2") as demo:
    gr.Markdown(
        "# Multimodal Quantum XAI App\n"
        "Enter a query (e.g., 'Simulate 6 qubit for 10 time units') or use sliders."
    )
    with gr.Row():
        query = gr.Textbox(
            label="Query", placeholder="e.g., Simulate 6 qubit for 10 time units")
    with gr.Row():
        num_qubits = gr.Slider(2, 6, value=2, label="Number of Qubits", step=1)
        max_time = gr.Slider(5, 15, value=10, label="Max Time")
        num_points = gr.Slider(50, 200, value=100, label="Time Points")
        epochs = gr.Slider(100, 2000, value=100, label="Training Epochs")
        decoherence_rate = gr.Slider(
            0.0, 0.1, value=0.0, label="Decoherence Rate")
    run_btn = gr.Button("Run Simulation")
    output_plot = gr.Plot(label="Results")
    output_text = gr.Textbox(label="Explanation")

    run_btn.click(
        fn=quantum_xai_app,
        inputs=[query, num_qubits, max_time,
                num_points, epochs, decoherence_rate],
        outputs=[output_plot, output_text]
    )

if __name__ == "__main__":
    demo.launch(share=False)
