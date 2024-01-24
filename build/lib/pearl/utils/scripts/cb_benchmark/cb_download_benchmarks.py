import os
import zipfile
from typing import Any, Dict

from pearl.utils.functional_utils.requests_get import requests_get

uci_urls = {
    "letter": {
        "url": "https://archive.ics.uci.edu/static/public/59",
        "file_name": "letter+recognition.zip",
    },
    "pendigits": {
        "url": "https://archive.ics.uci.edu/static/public/81",
        "file_name": "pen+based+recognition+of+handwritten+digits.zip",
    },
    "satimage": {
        "url": "https://archive.ics.uci.edu/static/public/146",
        "file_name": "statlog+landsat+satellite.zip",
    },
    "yeast": {
        "url": "https://archive.ics.uci.edu/static/public/110",
        "file_name": "yeast.zip",
    },
}


def download_uci_data(data_path: str) -> None:
    """
    Download UCI dataset, unzip, and delete the zip file.
    """

    for dataset_name in uci_urls.keys():
        url = os.path.join(
            uci_urls[dataset_name]["url"], uci_urls[dataset_name]["file_name"]
        )
        filename = os.path.join(data_path, uci_urls[dataset_name]["file_name"])

        # Download the zip file
        with open(filename, "wb") as f:
            f.write(requests_get(url).content)

        # Unzip the file
        unzip_filepath = os.path.join(data_path, dataset_name)
        with zipfile.ZipFile(filename, "r") as z:
            z.extractall(unzip_filepath)

        # Delete the zip file
        os.remove(filename)
