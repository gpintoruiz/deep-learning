import cv2
import os
from torch.utils.data import Dataset

from pathlib import Path
from carvana_unet.config import PROCESSED_DATA_DIR


# ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
images_path = PROCESSED_DATA_DIR / "train_images"
masks_path = PROCESSED_DATA_DIR / "train_masks"

class CarvanaDataset(Dataset):
    def __init__(self, image_dir=images_path, mask_dir=masks_path, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir) # Listamos todas las imagenes en la carpeta

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        """ Carga de la imagen """

        # Entramos a la carpeta y conseguimos la imagen de la lista
        img_path = os.path.join(self.image_dir, self.images[index])

        # Leemos la imagen y la pasamo a RGB
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        """ Carga de la máscara """

        # Entramos a la carpeta y conseguimos la mascara de la lista.
        # La razon por la que uso la misma lista de imagenes es porque la imagen y la mascara
        # tienen el mismo nombre, solo cambia la ruta de la carpeta y la extensión.
        """ image: data/train_images/0cdf5b5d0ce1_01.jpg """
        """ mask: data/train_masks/0cdf5b5d0ce1_01.png """
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))

        # Leemos la mascara en escala de grises
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Binarizamos la mascara
        mask[mask==255.0] = 1.0

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask
