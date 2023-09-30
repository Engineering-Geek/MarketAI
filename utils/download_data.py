from utils.S3 import S3Bucket
from tqdm import tqdm
import logging
import os
from pathlib import Path


def download(bucket_name: str, local_directory: Path, remote_directory: Path = None, verbosity: int = 2):
    """
    Download files from an S3 bucket to a local directory.

    Args:
        bucket_name (str): The name of the S3 bucket.
        local_directory (Path): The local directory to download files to.
        remote_directory (Path): The remote directory to download files from.
        verbosity (int, optional): Verbosity level for logging (0 = None, 1 = Errors, 2 = Progress, 3 = Debug).
                                   Defaults to 2.

    Returns:
        None
    """
    # Configure logging based on verbosity level
    if verbosity == 0:
        logging.disable(logging.CRITICAL)  # Disable all logging
    elif verbosity == 1:
        logging.basicConfig(level=logging.ERROR)  # Only show error messages
    elif verbosity == 3:
        logging.basicConfig(level=logging.DEBUG)  # Show debug messages

    bucket = S3Bucket(bucket_name)
    objects = bucket.list_directory(remote_directory) if remote_directory else bucket.list()

    for obj in tqdm(objects, desc="Downloading", disable=verbosity < 2):
        file = obj["Key"]
        if file.endswith("/"):
            continue
        path = Path("/".join(file.split("/")[:-1]))
        path = local_directory / path
        path.mkdir(parents=True, exist_ok=True)
        path /= file.split("/")[-1]
        try:
            df = bucket.get_dataframe(file)
            df.to_csv(path, index=False)
            if verbosity >= 2:
                logging.info(f'Downloaded: {file}')
        except Exception as e:
            if verbosity >= 1:
                logging.error(f'Error downloading {file}: {str(e)}')


if __name__ == "__main__":
    download(
        bucket_name="market-news-nm",
        local_directory=Path("/home/nikhil/PycharmProjects/TraderAI/MarketAI/data"),
        remote_directory=None,
        verbosity=2
    )
