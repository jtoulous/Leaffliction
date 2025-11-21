import gradio as gr

def tab_augmentation():
    gr.Markdown("""
        ## Augmentation

        Select the source folder, destination folder, type of augmentation, and other parameters. Click "Augment Images" to perform the augmentations and view sample results.
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
            augmentation_input = gr.Dropdown(label="Type of augmentation to process", value='All', choices=['All', 'Rotation', 'Blur', 'Contrast', 'Scaling', 'Illumination', 'Projective'], interactive=True)
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
                value="data/leaves",
                interactive=True
            )
            augment_button = gr.Button("Augment Images", variant="primary")
        with gr.Column():
            status = gr.Textbox(label="Status", interactive=False)
            status_md = gr.Markdown("### Sample Augmented Images")
            with gr.Row():
                image_original = gr.Image(label="") # 1
                image_blur = gr.Image(label="") # 3
                image_scaled = gr.Image(label="") # 5
            with gr.Row():
                image_illumination = gr.Image(label="") # 6
                image_contrast = gr.Image(label="") # 4
            with gr.Row():
                image_rotate = gr.Image(label="") # 2
                image_perspective = gr.Image(label="") # 7
    def display_augmentation(source, destination, range_percent, range_nb, seed, augmentation_type, display):
        import numpy as np
        import os
        from Augmentation import ImgAugmentator
        from srcs.tools import load_original_images, save_images, range_processing

        np.random.seed(seed if seed != 0 else None)

        if len(source) == 2 and not os.path.isdir(source[1]):
            source_path = source[-1]
        else:
            source_path = source[0]

        images, type_of_load = load_original_images(source_path)
        images = range_processing(images, range_percent=range_percent, range_nb=range_nb if range_nb != 0 else None)

        # Get a random image from the nested structure
        random_image = None
        if len(images) > 0:
            random_category = np.random.choice(list(images.keys()))
            if len(images[random_category]) > 0:
                random_key = np.random.choice(list(images[random_category].keys()))
                random_image = images[random_category][random_key]

        augmentator = ImgAugmentator(images)
        if type_of_load != 'File':
            augmentator.update_image_struct()

        if augmentation_type == 'All':
            augmentation_type = None

        augmented_images = augmentator.augment(augmentation=augmentation_type)

        save_images(augmented_images, destination)

        status_msg = f"Augmented {sum(len(imgs) for imgs in augmented_images.values())} images and saved to {destination}."

        random_augmented_images = []
        if display == 'Yes' and random_image is not None:
            for _, aug_img in augmented_images.items():
                for img_name, aug in aug_img.items():
                    if np.array_equal(aug['original'], random_image['original']):
                        random_augmented_images.append(aug)
                        img_name = img_name
                        break

        image_outputs = [None] * 7
        if len(random_augmented_images) > 0:
            aug_dict = random_augmented_images[0]
            idx = 0
            for aug_type in aug_dict.keys():
                if idx < 7:
                    if aug_type == 'original':
                        file_path = os.path.join(destination, random_category, f"{img_name.rstrip('.JPG')}.JPG")
                    else:
                        file_path = os.path.join(destination, random_category, f"{img_name.rstrip('.JPG')}_{aug_type}.JPG")

                    if os.path.exists(file_path):
                        image_outputs[idx] = gr.Image(value=file_path, label=aug_type, visible=True)
                    idx += 1

            if idx < 7:
                for j in range(idx, 7):
                    image_outputs[j] = gr.Image(value=None, label="", visible=False)

        status_md = f"### Sample Augmented Image: {img_name}" if len(random_augmented_images) > 0 else "### No augmented images to display."

        return status_msg, status_md, *image_outputs

    augment_button.click(
        display_augmentation,
        inputs=[source_input, dest_output, range_percent_input, range_nb_input, seed_input, augmentation_input, display_output],
        outputs=[status, status_md, image_original, image_rotate, image_blur, image_illumination, image_scaled, image_contrast, image_perspective]
    )
