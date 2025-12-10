import gradio as gr
import plotly.graph_objects as go

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


def tab_distribution():
    gr.Markdown("""
        ## Distribution

        Select the source folder, graph type, and whether to include all
        images or only original images. Click "Display Distribution" to
        generate the graphs.
    """)
    with gr.Row():
        with gr.Column(scale=0, min_width=400):
            distribution_input = gr.Radio(
                choices=['All', 'Pie Chart', 'Bar Plot'],
                label="Graph Type", value='All'
            )
            all_images_input = gr.Radio(
                choices=['Yes', 'No'],
                label="All Images (including augmented images)",
                value='No'
            )
            source_input = gr.FileExplorer(
                label="Source Folder",
                value=None,
                glob="**/*.JPG",
                root_dir="data/",
                ignore_glob="__pycache__"
            )
            display_button = gr.Button(
                "Display Distribution", variant="primary"
            )
        with gr.Column():
            status = gr.Textbox(
                label="Status",
                value="Waiting for distribution...",
                interactive=False
            )
            gr.Markdown("### Distribution Graphs")
            with gr.Row():
                pie_chart_output = gr.Plot(label="Pie Chart Output")
                bar_chart_output = gr.Plot(label="Bar Chart Output")

    def display_distribution(source, distribution_type, all_images):
        import pandas as pd
        import plotly.express as px
        from srcs.tools import load_images_from_list

        images = load_images_from_list(
            source,
            original=False if all_images == 'Yes' else True
        )

        distribution_type = (
            distribution_type.replace(' Chart', '')
            .replace(' Plot', '').lower()
        )

        categories = []
        counts = []
        for category, imgs in images.items():
            categories.append(category)
            counts.append(sum(len(variations) for variations in imgs.values()))

        pie_df = pd.DataFrame({
            'category': list(categories),
            'values': list(counts)
        })
        fig = px.pie(
            pie_df, values='values', names='category',
            color_discrete_sequence=[
                '#ff7c00', '#ff8c1a', '#ff9c33', '#ffac4d',
                '#ffbc66', '#ffcc80', '#ffdc99', '#ffecb3', '#fffbf0'
            ]
        )
        fig.update_layout(template=gradio_template)
        pie_chart = gr.Plot(
            value=fig,
            visible=True if distribution_type in ['all', 'pie'] else False
        )

        bar_df = pd.DataFrame({
            'Category': categories,
            'Count': counts
        })

        bar_fig = px.bar(
            bar_df, x='Category', y='Count',
            title='Image Distribution by Category',
            color='Category',
            color_discrete_sequence=[
                '#ff7c00', '#ff8c1a', '#ff9c33', '#ffac4d',
                '#ffbc66', '#ffcc80', '#ffdc99', '#ffecb3', '#fffbf0'
            ]
        )
        bar_fig.update_layout(
            template=gradio_template, showlegend=False
        )
        bar_image = gr.Plot(
            value=bar_fig,
            visible=True if distribution_type in ['all', 'bar'] else False
        )

        status_msg = "Distribution generated successfully."

        return status_msg, pie_chart, bar_image

    display_button.click(
        display_distribution,
        inputs=[source_input, distribution_input, all_images_input],
        outputs=[status, pie_chart_output, bar_chart_output]
    )
