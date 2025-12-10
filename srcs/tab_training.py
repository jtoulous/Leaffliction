import gradio as gr
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from srcs.tools import load_images_from_list
from srcs.DetectionAgent import DetectionAgent

# from tools import load_original_images, load_images
# from DetectionAgent import DetectionAgent
# import sys

# Custom Plotly template to match Gradio theme
gradio_template = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor='#27272a',
        plot_bgcolor='#27272a',
        font=dict(
            family=(
                'system-ui, -apple-system, BlinkMacSystemFont, '
                '"Segoe UI", Roboto, sans-serif'
            ),
            size=12,
            color='#ffffff'
        ),
        colorway=[
            '#ff7c00', '#ff8c1a', '#ff9c33', '#ffac4d', '#ffbc66',
            '#ffcc80', '#ffdc99', '#ffecb3', '#fffbf0'
        ],
        title=dict(
            font=dict(size=16, color='#ffffff'),
            x=0.5,
            xanchor='center'
        ),
        legend=dict(
            bgcolor='#18181b',
            bordercolor='#3f3f46',
            borderwidth=1,
            font=dict(color='#ffffff')
        )
    )
)


def tab_training():
    gr.Markdown("""
        ## Training

        Configure the training parameters and run the training.
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
            with gr.Column():
                gr.Markdown("### Training Parameters")
                epochs_input = gr.Number(
                    label="Epochs", value=10, precision=0
                )
                batch_size_input = gr.Number(
                    label="Batch Size", value=32, precision=0
                )
                range_test_input = gr.Slider(
                    minimum=0,
                    maximum=500,
                    step=1,
                    label="Number of images to exclude from training",
                    value=100,
                    interactive=True
                )
                seed_input = gr.Number(
                    label="Seed for splitting data", value=42, precision=0
                )

            # Save Parameters
            with gr.Column():
                gr.Markdown("### Save Parameters")
                save_folder_input = gr.Textbox(
                    label="Save Folder Path",
                    value="/",
                    placeholder="Ex: models/agent1"
                )
                save_name_input = gr.Textbox(
                    label="Agent Name", value="DetectionAgent_1"
                )

            train_button = gr.Button(
                "Run Training", variant="primary"
            )

        with gr.Column():
            training_status = gr.Textbox(
                label="Status",
                value="Waiting for training...",
                interactive=False
            )
            with gr.Row():
                training_results = gr.Plot(label="Results")

    train_button.click(
        RunTraining,
        inputs=[
            source_input,
            transformations_input,
            save_folder_input,
            save_name_input,
            epochs_input,
            batch_size_input,
            range_test_input,
            seed_input
        ],
        outputs=[training_status, training_results]
    )


def RunTraining(
    imgs_folder, transformations, save_folder, save_name,
    epochs, batch_size, range_test, seed
):
    try:
        from srcs.tools import range_processing

        save_path = os.path.join(save_folder, save_name)

        X = []
        y = []

        np.random.seed(seed)

        images = load_images_from_list(imgs_folder, original=False)
        total_images = sum(len(imgs) for imgs in images.values())
        range_total = (
            total_images - range_test
            if range_test < total_images
            else total_images
        )
        images = range_processing(images, range_nb=range_total)

        np.random.seed(None)

        for img_class, imgs_list in images.items():
            for img_name, img_types in imgs_list.items():
                for img_type, img in img_types.items():
                    X.append(img)
                    y.append(img_class)

        X = np.array(X)
        y = np.array(y)

        agent = DetectionAgent(
            epochs=epochs, batch_size=batch_size, transfo=transformations
        )
        history, test_accuracy, test_loss = agent.train(X, y)
        agent.save(save_path)

        # Generate plots
        epochs_range = list(range(1, len(history.history['accuracy']) + 1))

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Model Accuracy", "Model Loss")
        )

        # Accuracy
        fig.add_trace(
            go.Scatter(
                x=epochs_range, y=history.history['accuracy'],
                mode='lines+markers', name='Train Accuracy',
                line=dict(color='#ff7c00')
            ),
            row=1, col=1
        )
        if 'val_accuracy' in history.history:
            fig.add_trace(
                go.Scatter(
                    x=epochs_range, y=history.history['val_accuracy'],
                    mode='lines+markers', name='Validation Accuracy',
                    line=dict(color='#ff9c33')
                ),
                row=1, col=1
            )

        # Loss
        fig.add_trace(
            go.Scatter(
                x=epochs_range, y=history.history['loss'],
                mode='lines+markers', name='Train Loss',
                line=dict(color='#ffac4d')
            ),
            row=1, col=2
        )
        if 'val_loss' in history.history:
            fig.add_trace(
                go.Scatter(
                    x=epochs_range, y=history.history['val_loss'],
                    mode='lines+markers', name='Validation Loss',
                    line=dict(color='#ffbc66')
                ),
                row=1, col=2
            )

        fig.update_layout(template=gradio_template)
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=2)

        status_message = (
            f"Training Successful!\n"
            f"Agent saved at: {save_path}\n"
            f"--------------------------------\n"
            f"Final Validation Accuracy: {test_accuracy:.2%}\n"
            f"Final Validation Loss: {test_loss:.4f}"
        )

        return status_message, fig

    except Exception as e:
        return f'Error during training: {str(e)}', None


# X = []
# y = []
#
# original_images, type_of_load = load_images(sys.argv[1])
# breakpoint()
#
# for img_class, imgs_list in original_images.items():
#     for img_name, img_types in imgs_list.items():
#         for img_type, img in img_types.items():
#             X.append(img)
#             y.append(img_class)
#
# breakpoint()
#
# X = np.array(X)
# y = np.array(y)
