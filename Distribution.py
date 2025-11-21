import numpy as np
import argparse as ap
import seaborn as sns
import matplotlib.pyplot as plt

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from srcs.tools import load_original_images, load_images


class Distribution:
    def __init__(self, images_structure, all_images=False):
        """
        Initialize the Distribution with a given structure of images.

        Args:
            images_structure (dict): A dictionary containing images categorized by class names.
        """
        self.images_structure = images_structure

    def distribution(self, progress=None, task=None, graph_type=None):
        """
        Calculate the distribution of images across different classes.

        Args:
            progress (Progress, optional): Rich Progress object for displaying progress.
            task (Task, optional): Specific task in the progress to update.
            graph_type (str, optional): Type of graph to display ('pie', 'bar').

        Returns:
            dict: A dictionary where keys are class names and values are the count of images in each class.
        """
        self._pie_chart() if graph_type in ['pie', None] else None
        if task is not None:
            progress.update(task, advance=1)

        self._bar_plot() if graph_type in ['bar', None] else None
        if task is not None:
            progress.update(task, advance=1)

    def get_distribution_graphs(self, graph_type=None):
        """
        Generate distribution graphs based on the specified type.

        Args:
            graph_type (str, optional): Type of graph to generate ('pie', 'bar', 'all').

        Returns:
            tuple: A tuple containing the pie chart and bar chart images.
        """
        pie_chart = None
        bar_chart = None

        if graph_type in ['pie', 'all', None]:
            pie_chart = self._generate_pie_chart_image()

        if graph_type in ['bar', 'all', None]:
            bar_chart = self._generate_bar_plot_image()

        return pie_chart, bar_chart

    def _pie_chart(self):
        """
        Generate a pie chart representing the distribution of images across classes.
        """
        data = []
        labels = []
        chart = {}

        for category, images in self.images_structure.items():
            chart[category] = sum(len(variations) for variations in images.values())

        chart = dict(sorted(chart.items(), key=lambda item: item[1], reverse=False))
        data = list(chart.values())
        labels = list(chart.keys())

        colors = sns.color_palette('light:r_r', len(labels))

        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(data, labels=labels, colors=colors, autopct='%.0f%%')

        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w", alpha=0.9),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def on_hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                for i, wedge in enumerate(wedges):
                    if wedge.contains(event)[0]:
                        annot.xy = (event.xdata, event.ydata)
                        text = f"{labels[i]}\nCount: {data[i]}"
                        annot.set_text(text)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        return
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_hover)
        plt.show()

    def _generate_pie_chart_image(self):
        """
        Generate a pie chart image as a numpy array.

        Returns:
            np.ndarray: The pie chart as an image array.
        """
        data = []
        labels = []
        chart = {}

        for category, images in self.images_structure.items():
            chart[category] = sum(len(variations) for variations in images.values())

        chart = dict(sorted(chart.items(), key=lambda item: item[1], reverse=False))
        data = list(chart.values())
        labels = list(chart.keys())

        colors = sns.color_palette('light:r_r', len(labels))

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.pie(data, labels=labels, colors=colors, autopct='%.0f%%')

        # Convert to image array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB
        image = image[:, :, :3]

        plt.close(fig)
        return image

    def _bar_plot(self):
        """
        Generate a bar plot representing the distribution of images across classes.
        """
        data = []
        labels = []
        chart = {}

        for category, images in self.images_structure.items():
            chart[category] = sum(len(variations) for variations in images.values())

        chart = dict(sorted(chart.items(), key=lambda item: item[0]))
        data = list(chart.values())
        labels = list(chart.keys())

        sorted_indices = np.argsort(data)
        colors = sns.color_palette('light:r_r', len(labels))
        color_map = [colors[np.where(sorted_indices == i)[0][0]] for i in range(len(data))]

        fig, ax = plt.subplots()
        bars = ax.bar(labels, data, color=color_map)

        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        plt.xticks(rotation=45)

        annot = ax.annotate("", xy=(0, 0), xytext=(-30, -40), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w", alpha=0.9),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def on_hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                for i, bar in enumerate(bars):
                    if bar.contains(event)[0]:
                        x = bar.get_x() + bar.get_width() / 2
                        y = bar.get_height()
                        annot.xy = (x, y)
                        text = f"{labels[i]}\nCount: {data[i]}"
                        annot.set_text(text)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        return
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_hover)

        plt.tight_layout()
        plt.show()

    def _generate_bar_plot_image(self):
        """
        Generate a bar plot image as a numpy array.

        Returns:
            np.ndarray: The bar plot as an image array.
        """
        data = []
        labels = []
        chart = {}

        for category, images in self.images_structure.items():
            chart[category] = sum(len(variations) for variations in images.values())

        chart = dict(sorted(chart.items(), key=lambda item: item[0]))
        data = list(chart.values())
        labels = list(chart.keys())

        sorted_indices = np.argsort(data)
        colors = sns.color_palette('light:r_r', len(labels))
        color_map = [colors[np.where(sorted_indices == i)[0][0]] for i in range(len(data))]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(labels, data, color=color_map)

        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        plt.xticks(rotation=45)
        plt.tight_layout()

        # Convert to image array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB
        image = image[:, :, :3]

        plt.close(fig)
        return image


def ArgumentParsing():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '--source',
        type=str,
        default='data/leaves',
        help='Folder with original images (default: data/leaves)')
    parser.add_argument(
        '--distribution',
        type=str,
        choices=['pie', 'bar'],
        default=None,
        help='Distribution to display (default: None)')
    parser.add_argument(
        '--all-images',
        action='store_true',
        help='Display distribution for all images (not only original) (default: False)')

    return parser.parse_args()


if __name__ == '__main__':
    try:
        args = ArgumentParsing()

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.completed}/{task.total}",
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
        ) as progress:
            global_task = progress.add_task("Global Progress", total=2)

            # Load images
            images_load_task = progress.add_task("↪ Load images", total=0)
            if args.all_images:
                images, _ = load_images(args.source, progress=progress, task=images_load_task)
            else:
                images, _ = load_original_images(args.source, progress=progress, task=images_load_task)
            progress.update(global_task, advance=1)

            # Augment images
            display_task = progress.add_task("↪ Display distribution", total=2)
            Distributor = Distribution(images, all_images=args.all_images)
            Distributor.distribution(progress=progress, task=display_task, graph_type=args.distribution)
            progress.update(global_task, advance=1)

    except Exception as error:
        print(f'Error: {error}')
