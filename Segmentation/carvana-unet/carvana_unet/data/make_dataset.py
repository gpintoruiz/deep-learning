import kaggle
from carvana_unet.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
import os
from zipfile import ZipFile

from loguru import logger


dataset_name = "ipythonx/carvana-image-masking-png"

# Download the dataset
logger.info("Downloading dataset...")
kaggle.api.dataset_download_files(dataset_name, RAW_DATA_DIR)

# Unzip the downloaded files
zip_file_path = os.path.join(RAW_DATA_DIR, f'{dataset_name.split("/")[-1]}.zip')
with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(PROCESSED_DATA_DIR)

logger.success(f'Data for {dataset_name} downloaded and extracted to {PROCESSED_DATA_DIR}.')