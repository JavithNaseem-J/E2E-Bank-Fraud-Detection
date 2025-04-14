import os
import zipfile
from urllib import request
from pathlib import Path
from project import logger
from project.utils import get_size
from project.entity.config_entity import DataIngestionConfig



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    def download_file(self):
        if not os.path.exists(self.config.zip_file):
            filename, headers = request.urlretrieve(
            url = self.config.source_url,
            filename = self.config.zip_file)
            logger.info(f'{filename} downlod with following information: \n{headers}')
        else:
            logger.info(f"File already exixts of the size: {get_size(Path(self.config.zip_file))}")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_file
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.zip_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)