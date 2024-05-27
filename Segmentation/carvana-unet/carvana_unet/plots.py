from pathlib import Path
import matplotlib.pyplot as plt
from carvana_unet.config import FIGURES_DIR

def visualize_data(dataset, name_plot: str, output_path=FIGURES_DIR, size=4):
    plt.figure(figsize=(8, 10))
    for i in range(1, size*2 + 1, 2): # 1, 3, 5, 7
        img = dataset[i][0]
        mask = dataset[i][1]
        plt.subplot(size, 2, i); plt.imshow(img)
        plt.subplot(size, 2, i + 1); plt.imshow(mask, cmap='gray')
    plt.savefig(f"{output_path}/{name_plot}")
