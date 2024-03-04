# pyre-strict
import os
import zipfile

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
        unzipped_dataset_dirpath = os.path.join(data_path, dataset_name)
        zip_filepath = os.path.join(data_path, uci_urls[dataset_name]["file_name"])

        # Download the zip file
        url = uci_urls[dataset_name]["url"] + "/" + uci_urls[dataset_name]["file_name"]
        response = requests_get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to download {dataset_name} dataset from {url}.")

        # Locally save the zip file
        with open(zip_filepath, "wb") as f:
            f.write(response.content)

        # Unzip the file
        try:
            with zipfile.ZipFile(zip_filepath, "r") as z:
                z.extractall(unzipped_dataset_dirpath)
        except zipfile.BadZipFile:
            raise zipfile.BadZipFile(
                f"Bad zip file: {zip_filepath}. Please delete corrupt file and run again."
            )

        # Delete the zip file
        os.remove(zip_filepath)
