import matplotlib.pyplot as plt
import numpy as np

def DisplayImages(img_list, title_list):
    img_list = [np.array(img) if not isinstance(img, np.ndarray) else img for img in img_list]
    fig, axes = plt.subplots(1, len(img_list), figsize=(20, 20))
    for ax, img, title in zip(axes, img_list, title_list):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.show()