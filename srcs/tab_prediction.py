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
                value=None,
                glob="**/*.JPG",
                root_dir="data/",
                ignore_glob="__pycache__"
            )
            use_range_checkbox = gr.Checkbox(label="Use range to limit number of images", value=True)
            range_test_input = gr.Slider(
                minimum=0,
                maximum=500,
                step=1,
                label="Number of images that have been excluded from training",
                value=100,
                interactive=True,
                visible=True
            )
            seed_input = gr.Number(label="Seed that was used for splitting data", value=42, precision=0, visible=True)

            agent_input = gr.File(
                label="Agent Folder",
                file_count="directory"  # PERMET DE DRAG & DROP UN DOSSIER
            )
            prediction_button = gr.Button("Run Prediction", variant="primary")

        with gr.Column():
            prediction_accuracy_status = gr.Textbox(label="Status", value="Waiting for prediction...", interactive=False)
            with gr.Row():
                img_output = gr.Gallery(label="Transformed Images", columns=4)

    def update_range_visibility(use_range):
        return gr.update(visible=use_range), gr.update(visible=use_range)

    use_range_checkbox.change(
        update_range_visibility,
        inputs=[use_range_checkbox],
        outputs=[range_test_input, seed_input]
    )

    prediction_button.click(
        run_prediction,
        inputs=[source_input, agent_input, range_test_input, seed_input, use_range_checkbox],
        outputs=[prediction_accuracy_status, img_output]
    )


def run_prediction(source_img, agent_folder, range_test, seed, use_range):

    # ---------------------------
    #   SOURCE IMAGE
    # ---------------------------
    from srcs.tools import load_images_from_list

    images = load_images_from_list(source_img)

    if use_range:
        import numpy as np
        from srcs.tools import range_processing

        np.random.seed(seed)
        images = range_processing(images, range_nb=range_test)
        np.random.seed(None)

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

    if model_weights_file is None or agent_file is None or model_arch_file is None:
        raise gr.Error("Missing agent files. Please ensure model.weights.h5, model_architecture.json, and agent.pkl are present.")

    # ---------------------------
    #   RUN PRED
    # ---------------------------
    agent = DetectionAgent.load_from_files(model_weights_file, agent_file, model_arch_file)

    predictions_list = []
    gallery_images = []
    correct_count = 0
    total_count = 0

    for cat, cat_imgs in images.items():
        for img_name, img_variants in cat_imgs.items():
            # Use 'original' if available, otherwise first available
            target_img_bgr = img_variants.get('original')
            if target_img_bgr is None:
                if len(img_variants) > 0:
                    target_img_bgr = list(img_variants.values())[0]
                else:
                    continue

            # Predict (expects BGR)
            prediction, transformed_imgs_bgr = agent.predict(target_img_bgr)

            # Accuracy
            is_correct = (prediction == cat)
            if is_correct:
                correct_count += 1
            total_count += 1

            icon = "✅" if is_correct else "❌"
            predictions_list.append(f"{icon} {img_name}: {prediction} (True: {cat})")

            # Border color (Green if correct, Red if wrong)
            border_color = (0, 255, 0) if is_correct else (0, 0, 255)

            # Add original to gallery (convert to RGB)
            img_border = cv2.copyMakeBorder(
                target_img_bgr, 10, 10, 10, 10,
                cv2.BORDER_CONSTANT, value=border_color
            )
            gallery_images.append((cv2.cvtColor(img_border, cv2.COLOR_BGR2RGB), f"{img_name} | Pred: {prediction} | True: {cat}"))

            # Add transformed images to gallery (convert to RGB)
            for i, t_img in enumerate(transformed_imgs_bgr):
                t_img_border = cv2.copyMakeBorder(
                    t_img, 10, 10, 10, 10,
                    cv2.BORDER_CONSTANT, value=border_color
                )
                label = agent.transformations[i] if i < len(agent.transformations) else f"Transfo {i+1}"
                gallery_images.append((cv2.cvtColor(t_img_border, cv2.COLOR_BGR2RGB), f"{label} | Pred: {prediction} | True: {cat}"))

    if total_count > 1:
        # Multiple images: show accuracy
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        accuracy_status = f"Global Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})"
    else:
        # Single image: just show prediction
        accuracy_status = "\n".join(predictions_list)

    return accuracy_status, gallery_images
