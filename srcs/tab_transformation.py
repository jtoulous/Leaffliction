import gradio as gr

def tab_transformation():
    gr.Markdown("""
        ## Transformation

        Select the source folder, graph type, and whether to include all images or only original images. Click "Display Distribution" to generate the graphs.
    """)
    with gr.Row():
        with gr.Column(scale=0, min_width=400):
            range_percent_input = gr.Slider(
                minimum=1,
                maximum=100,
                step=1,
                label="Percentage of image to process",
                value=100,
                interactive=True
            )
            range_nb_input = gr.Number(label="Number of image to process (0 for all)", value=0, precision=0, interactive=True)
            seed_input = gr.Number(label="Random seed (0 for random)", value=0, precision=0, interactive=True)
            transformation_input = gr.Dropdown(label="Type of transformation to process", value='All', choices=['All', 'Gaussian blur', 'Mask', 'Roi objects', 'Pseudolandmarks', 'Spots isolation', 'Background removal'], interactive=True)
            display_output = gr.Radio(choices=['Yes', 'No'], label="Display image augmentation", value='Yes')
            source_input = gr.FileExplorer(
                label="Source Folder",
                value="leaves",
                glob="**/*",
                root_dir="data/",
                ignore_glob="__pycache__"
            )
            dest_output = gr.Textbox(
                label="Destination Folder",
                value="data/leaves_preprocessed",
                interactive=True
            )
            transformation_button = gr.Button("Transform Images", variant="primary")
        with gr.Column():
            status = gr.Textbox(label="Status", interactive=False)
            status_md = gr.Markdown("### Sample Transformed Images")
            with gr.Row():
                image_output_1 = gr.Image(label="")
                image_output_3 = gr.Image(label="")
                image_output_5 = gr.Image(label="")
            with gr.Row():
                image_output_2 = gr.Image(label="")
                image_output_4 = gr.Image(label="")
            with gr.Row():
                image_output_6 = gr.Image(label="")
                image_output_7 = gr.Image(label="")

    def display_transformation(source, destination, range_percent, range_nb, seed, transformation_type, display):
        import numpy as np
        import os
        from Transformation import ImgTransformator
        from srcs.tools import load_images, load_original_images, save_images, range_processing

        np.random.seed(seed if seed != 0 else None)

        if len(source) == 2 and not os.path.isdir(source[1]):
            source_path = source[-1]
        else:
            source_path = source[0]

        images, type_of_load = load_images(source_path)
        images = range_processing(images, range_percent=range_percent, range_nb=range_nb if range_nb != 0 else None)

        # Get a random image from the nested structure
        random_image = None
        if len(images) > 0:
            random_category = np.random.choice(list(images.keys()))
            if len(images[random_category]) > 0:
                random_key = np.random.choice(list(images[random_category].keys()))
                random_image = images[random_category][random_key]

        transformator = ImgTransformator(images)

        if transformation_type == 'All':
            transformation_type = None

        transformed_images = transformator.transform(transform=transformation_type)

        save_images(transformed_images, destination)

        status_msg = f"Transformed {sum(len(imgs) for imgs in transformed_images.values())} images and saved to {destination}."

        random_transformed_images = []
        if display == 'Yes' and random_image is not None:
            for _, trans_img in transformed_images.items():
                for img_name, trans in trans_img.items():
                    if np.array_equal(trans['original'], random_image['original']):
                        random_transformed_images.append(trans)
                        img_name = img_name
                        break

        image_outputs = [None] * 7
        if len(random_transformed_images) > 0:
            trans_dict = random_transformed_images[0]
            idx = 0
            for trans_type, img_array in trans_dict.items():
                if idx < 7:
                    if trans_type == 'original':
                        file_path = os.path.join(destination, random_category, f"{img_name.rstrip('.JPG')}.JPG")
                    else:
                        file_path = os.path.join(destination, random_category, f"{img_name.rstrip('.JPG')}_{trans_type}.JPG")

                    if os.path.exists(file_path):
                        image_outputs[idx] = gr.Image(value=file_path, label=trans_type, visible=True)
                    idx += 1

            if idx < 7:
                for j in range(idx, 7):
                    image_outputs[j] = gr.Image(value=None, label="", visible=False)

        status_md = f"### Sample Transformed Image: {img_name}" if len(random_transformed_images) > 0 else "### No transformed images to display."

        return status_msg, status_md, *image_outputs

    transformation_button.click(
        display_transformation,
        inputs=[source_input, dest_output, range_percent_input, range_nb_input, seed_input, transformation_input, display_output],
        outputs=[status, status_md, image_output_1, image_output_2, image_output_3, image_output_4, image_output_5, image_output_6, image_output_7]
    )
