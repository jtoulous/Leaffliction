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
        self.images_structure = {}

        if all_images:
            for category, images in images_structure.items():
                self.images_structure[category] = {}
                for img_key, variations in images.items():
                    for variation_name in variations.keys():
                        combined_key = f"{img_key}_{variation_name}"
                        self.images_structure[category][combined_key] = variations[variation_name]
        else:
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

    def _pie_chart(self):
        """
        Generate a pie chart representing the distribution of images across classes.
        """
        data = []
        labels = []
        chart = {}

        for category, images in self.images_structure.items():
            chart[category] = len(images)

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

    def _bar_plot(self):
        """
        Generate a bar plot representing the distribution of images across classes.
        """
        data = []
        labels = []
        chart = {}

        for category, images in self.images_structure.items():
            chart[category] = len(images)

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


def ArgumentParsing():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '--load-folder',
        type=str,
        default='data/leaves',
        help='Folder with original images (default: data/leaves)')
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)')
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
                images = load_images(args.load_folder, progress=progress, task=images_load_task)
            else:
                images = load_original_images(args.load_folder, progress=progress, task=images_load_task)
            progress.update(global_task, advance=1)

            np.random.seed(args.seed)

            # Augment images
            display_task = progress.add_task("↪ Display distribution", total=2)
            Distributor = Distribution(images, all_images=args.all_images)
            Distributor.distribution(progress=progress, task=display_task, graph_type=args.distribution)
            progress.update(global_task, advance=1)

    except Exception as error:
        print(f'Error: {error}')
