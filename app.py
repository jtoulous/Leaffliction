import gradio as gr

from srcs.tab_distribution import tab_distribution
from srcs.tab_augmentation import tab_augmentation
from srcs.tab_transformation import tab_transformation
from srcs.tab_training import tab_training
from srcs.tab_prediction import tab_prediction

css = """
* {
    scrollbar-color: white var(--neutral-800);
    scrollbar-width: thin;
}
input[type=number]::-webkit-inner-spin-button {
    -webkit-appearance: none;
}
"""

with gr.Blocks(theme="default", css=css) as demo:
    gr.Markdown("# Leaffliction 🌿🍂")
    with gr.Tab("Home"):
        with open("README.md", "r") as f:
            readme_content = f.read()
        readme_content = readme_content.lstrip("# Leaffliction 🌿🍂\n")
        gr.Markdown("## Welcome 👋 !\nFeel free to explore the tabs.\n")
        with gr.Accordion("📖 Readme", open=True)   :
            gr.Markdown(readme_content)
    with gr.Tab("Distribution"):
        tab_distribution()
    with gr.Tab("Augmentation"):
        tab_augmentation()
    with gr.Tab("Transformation"):
        tab_transformation()
    with gr.Tab("Training"):
        tab_training()
    with gr.Tab("Prediction"):
        tab_prediction()

if __name__ == "__main__":
    demo.launch()
