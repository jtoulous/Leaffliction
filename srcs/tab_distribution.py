import gradio as gr

def tab_distribution():
    gr.Markdown("""
        ## Distribution

        Select the source folder, graph type, and whether to include all images or only original images. Click "Display Distribution" to generate the graphs.
    """)
    with gr.Row():
        with gr.Column(scale=0, min_width=400):
            distribution_input = gr.Radio(choices=['All', 'Pie Chart', 'Bar Plot'], label="Graph Type", value='All')
            all_images_input = gr.Radio(choices=['Yes', 'No'], label="All Images (including augmented images)", value='No')
            source_input = gr.FileExplorer(
                label="Source Folder",
                value="leaves",
                glob="**/*",
                root_dir="data/",
                ignore_glob="__pycache__"
            )
            display_button = gr.Button("Display Distribution", variant="primary")
        with gr.Column():
            status = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                pie_chart_output = gr.Image(label="Pie Chart Output")
                bar_chart_output = gr.Image(label="Bar Chart Output")

    def display_distribution(source, distribution_type, all_images):
        import numpy as np
        from Distribution import Distribution
        from srcs.tools import load_images, load_original_images

        if all_images == 'Yes':
            images, _ = load_images(source[0])
        else:
            images, _ = load_original_images(source[0])

        distribution_type = distribution_type.replace(' Chart', '').replace(' Plot', '').lower()

        Distributor = Distribution(images, all_images=all_images == 'Yes')
        pie_chart, bar_chart = Distributor.get_distribution_graphs(graph_type=distribution_type)

        pie_image = gr.Image(value=pie_chart, visible=True if distribution_type in ['all', 'pie'] else False)
        bar_image = gr.Image(value=bar_chart, visible=True if distribution_type in ['all', 'bar'] else False)

        return pie_image, bar_image

    display_button.click(
        display_distribution,
        inputs=[source_input, distribution_input, all_images_input],
        outputs=[pie_chart_output, bar_chart_output]
    )
