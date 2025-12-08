import gradio as gr
import os
import cv2
from srcs.DetectionAgent import DetectionAgent


def tab_prediction():
    gr.Markdown("""
        ## Prediction

        Select the images, and click 'Run Predictions'.
    """)
    with gr.Row():

        with gr.Column(scale=0, min_width=400):
            source_input = gr.FileExplorer(
                label="Source Folder",
                value="leaves",
                glob="**/*.JPG",
                root_dir="data/",
                ignore_glob="__pycache__",
                file_count="single"
            )
            agent_input = gr.File(
                label="Agent Folder",
                file_count="directory"  # PERMET DE DRAG & DROP UN DOSSIER
            )
            prediction_button = gr.Button("Run Prediction", variant="primary")

        with gr.Column():
            prediction = gr.Textbox(label="Prediction", interactive=False)
            with gr.Row():
                img_output = gr.Image(label="original_img")

    prediction_button.click(
        run_prediction,
        inputs=[source_input, agent_input],
        outputs=[prediction, img_output]
    )


def run_prediction(source_img, agent_folder):

    # ---------------------------
    #   SOURCE IMAGE
    # ---------------------------
    # FileExplorer peut renvoyer : str, dict, list
    if isinstance(source_img, str):
        source_path = source_img
    elif isinstance(source_img, dict):
        source_path = source_img.get("path") or source_img.get("name")
    elif isinstance(source_img, list) and len(source_img) > 0:
        item = source_img[0]
        source_path = item.get("path") if isinstance(item, dict) else item
    else:
        raise ValueError(f"Invalid source image input: {source_img}")

    img = cv2.imread(source_path)
    if img is None:
        raise ValueError(f"cv2.imread failed for {source_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------------------------
    #   AGENT FILES
    # ---------------------------
    if not isinstance(agent_folder, list) or len(agent_folder) == 0:
        raise ValueError("Invalid agent folder input")

    model_weights_file = None
    model_arch_file = None
    agent_file = None

    for f in agent_folder:
        name = os.path.basename(f.name)
        if name == "model.weights.h5":
            model_weights_file = f.name
        elif name == "agent.pkl":
            agent_file = f.name

        elif name == "model_architecture.json":
            model_arch_file = f.name

    if model_weights_file is None or agent_file is None:
        raise ValueError("model.keras and agent.pkl not found in uploaded folder")

    # ---------------------------
    #   RUN PRED
    # ---------------------------
    agent = DetectionAgent.load_from_files(model_weights_file, agent_file, model_arch_file)
    prediction, transformed_imgs = agent.predict(img)

    return prediction, img
