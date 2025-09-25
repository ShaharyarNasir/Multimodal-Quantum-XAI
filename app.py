import gradio as gr
import torch
from src.simulate import simulate_quantum
from src.nn_model import train_nn
from src.shap_analysis import generate_shap_and_plot
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np
import os
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    tokenizer_path = os.path.join("data", "tokenizer")
    model_path = os.path.join("data", "model")
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    logger.info("Loaded DistilBERT model and tokenizer from %s, %s", tokenizer_path, model_path)
except Exception as e:
    logger.error("Failed to load model/tokenizer: %s", str(e))
    raise

def parse_query(query):
    try:
        logger.debug("Parsing query: %s", query)
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=16)
        logger.debug("Tokenized inputs: %s", inputs)
        with torch.no_grad():
            outputs = model(**inputs).logits
        params = outputs.squeeze().numpy()
        logger.debug("Raw model outputs: %s", params)
        parsed_params = {
            "num_qubits": min(max(int(round(params[0])), 2), 6),
            "max_time": min(max(int(round(params[1])), 5), 15),
            "num_points": int(round(params[2])) if 50 <= params[2] <= 200 else 100,
            "epochs": min(max(int(round(params[3])), 100), 2000),
            "decoherence_rate": min(max(float(params[4]), 0.0), 0.1)
        }
        logger.info("Parsed params: %s", parsed_params)
        return parsed_params
    except Exception as e:
        logger.error("Parsing error for query '%s': %s", query, str(e))
        return None

def quantum_xai_app(query, num_qubits, max_time, num_points, epochs, decoherence_rate):
    try:
        params = {
            "num_qubits": num_qubits,
            "max_time": max_time,
            "num_points": num_points,
            "epochs": epochs,
            "decoherence_rate": decoherence_rate
        }
        logger.debug("Slider params: %s", params)
        if query.strip():
            parsed_params = parse_query(query)
            if parsed_params is not None:
                params = parsed_params
                logger.info("Using parsed query params: %s", params)
            else:
                logger.warning("Query parsing failed for '%s', using slider defaults: %s", query, params)
        else:
            logger.info("No query provided, using slider params: %s", params)

        logger.info("Running simulation with params: %s", params)
        times, quantum_data = simulate_quantum(params)
        model, predicted, losses, final_mse = train_nn(times, quantum_data, params["epochs"])
        fig = generate_shap_and_plot(model, times, quantum_data, predicted, losses, final_mse,
                                     params["num_qubits"], params["decoherence_rate"])
        if fig is None:
            logger.error("SHAP plot generation failed")
            return None, f"Error: Failed to generate SHAP plot for {params['num_qubits']} qubits"
        explanation = f"Simulation for {params['num_qubits']} qubits, time 0-{params['max_time']}. MSE: {final_mse:.4f}"
        return fig, explanation
    except Exception as e:
        logger.error("Simulation error: %s", str(e))
        return None, f"Error: {str(e)}"

with gr.Blocks(title="Multimodal Quantum XAI App v2") as demo:
    gr.Markdown(
        "# Multimodal Quantum XAI App\n"
        "Enter a query (e.g., 'Simulate 6 qubit for 10 time units') or use sliders."
    )
    with gr.Row():
        query = gr.Textbox(label="Query", placeholder="e.g., Simulate 6 qubit for 10 time units")
    with gr.Row():
        num_qubits = gr.Slider(2, 6, value=2, label="Number of Qubits", step=1)
        max_time = gr.Slider(5, 15, value=10, label="Max Time")
        num_points = gr.Slider(50, 200, value=100, label="Time Points")
        epochs = gr.Slider(100, 2000, value=100, label="Training Epochs")
        decoherence_rate = gr.Slider(0.0, 0.1, value=0.0, label="Decoherence Rate")
    run_btn = gr.Button("Run Simulation")
    output_plot = gr.Plot(label="Results")
    output_text = gr.Textbox(label="Explanation")

    run_btn.click(
        fn=quantum_xai_app,
        inputs=[query, num_qubits, max_time, num_points, epochs, decoherence_rate],
        outputs=[output_plot, output_text]
    )

if __name__ == "__main__":
    demo.launch(share=False)