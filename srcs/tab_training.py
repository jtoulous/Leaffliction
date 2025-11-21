import gradio as gr
import os
import numpy as np
from srcs.tools import load_original_images
from srcs.DetectionAgent import DetectionAgent

def tab_training():
    gr.Markdown("""
        ## Training

        Configure the training parameters and run the training.
    """)
    with gr.Row():
        with gr.Column(scale=0, min_width=400):
            source_input = gr.FileExplorer(
                label="Source Folder",
                value="leaves",
                glob="**/*",
                root_dir="data/",
                ignore_glob="__pycache__"
            )
            transformations_input = gr.CheckboxGroup(
                choices=[
                    'gaussian_blur',
                    'mask', 
                    'roi_objects',
                    'pseudolandmarks',
                    'spots_isolation',
                    'background_removal'
                ],
                label="Transformations",
                value=['gaussian_blur']
            )
            
            # Training Parameters
            with gr.Group():
                gr.Markdown("### Training Parameters")
                epochs_input = gr.Number(label="Epochs", value=10, precision=0)
                batch_size_input = gr.Number(label="Batch Size", value=32, precision=0)
            
            # Save Parameters  
            with gr.Group():
                gr.Markdown("### Save Parameters")
                save_folder_input = gr.Textbox(
                    label="Save Folder Path",
                    value="/",
                    placeholder="Ex: models/agent1"
                )
                save_name_input = gr.Textbox(label="Agent Name", value="DetectionAgent_1")
            
            train_button = gr.Button("Run Training", variant="primary")
        
        with gr.Column():
            status = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                results_output = gr.Image(label="Results")

    train_button.click(
        RunTraining,
        inputs=[
            source_input, 
            transformations_input,
            save_folder_input,
            save_name_input,
            epochs_input,
            batch_size_input
        ],
        outputs=[status]
    )


def RunTraining(imgs_folder, transformations, save_folder, save_name, epochs, batch_size):
    try:
        save_path = os.path.join(save_folder, save_name)
        
        X = []
        y = []

        if len(imgs_folder) == 2 and not os.path.isdir(imgs_folder[1]):
            source_path = imgs_folder[-1]
        else:
            source_path = imgs_folder[0]

        original_images, type_of_load = load_original_images(source_path)
        for img_class, imgs_list in original_images.items():
            for img_name, img_types in imgs_list.items():
                for img_type, img in img_types.items():
                    X.append(img)
                    y.append(img_class)

        X = np.array(X)
        y = np.array(y)

        agent = DetectionAgent(epochs=epochs, batch_size=batch_size, transfo=transformations)
        agent.train(X, y)
        agent.save(save_path)

        return 'Training Successful'
    
    except Exception as e:
        return f'Error during training: {str(e)}'