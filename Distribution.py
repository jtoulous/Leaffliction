import numpy as np
import argparse as ap
import seaborn as sns
import matplotlib.pyplot as plt

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from srcs.tools import load_original_images


class Distribution:
    def __init__(self, images_structure):
        """
        Initialize the Distribution with a given structure of images.

        Args:
            images_structure (dict): A dictionary containing images categorized by class names.
        """
        self.images_structure = images_structure

        return

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
        plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
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

        ax = sns.barplot(x=labels, y=data, hue=labels, palette=color_map, legend=False)

        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        plt.xticks(rotation=45)

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
        '--range-nb',
        type=int,
        default=None,
        help='Number of images to process (default: None)')
    parser.add_argument(
        '--range-percent',
        type=int,
        default=100,
        help='Percentage of images to process (default: 100)')
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

    return parser.parse_args()


def range_processing(images, range_nb=None, range_percent=100):
    """
    Limit the number of images to process based on specified number and/or percentage.

    Args:
        images (dict): Dictionary of images categorized by class names.
        range_nb (int, optional): Maximum number of images to process. If None, no limit is applied.
        range_percent (int, optional): Percentage of images to process (0-100). Default is 100.

    Returns:
        dict: Dictionary of images limited to the specified number/percentage.

    Behavior:
        - Flattens the images dictionary into a list of (category, image_key, image) tuples.
        - Shuffles the list randomly.
        - Selects the first 'range_nb' images if specified.
        - Further limits the selection to 'range_percent' of the total images.
        - Reconstructs and returns a dictionary of the selected images.
    """
    all_images = [(cat, img_key, img) for cat, imgs in images.items() for img_key, img in imgs.items()]
    np.random.shuffle(all_images)
    all_images = all_images[:range_nb] if range_nb is not None else all_images

    limit = int(len(all_images) * range_percent / 100)
    all_images = all_images[:limit]

    images = {}
    for cat, img_key, img in all_images:
        if cat not in images:
            images[cat] = {}
        images[cat][img_key] = img

    return images


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
            images = load_original_images(args.load_folder, progress=progress, task=images_load_task)
            images = range_processing(images, range_nb=args.range_nb, range_percent=args.range_percent)
            progress.update(global_task, advance=1)

            np.random.seed(args.seed)

            # Augment images
            display_task = progress.add_task("↪ Display distribution", total=2)
            Distributor = Distribution(images)
            Distributor.distribution(progress=progress, task=display_task, graph_type=args.distribution)
            progress.update(global_task, advance=1)

    except Exception as error:
        print(f'Error: {error}')
