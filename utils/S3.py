import logging
from pathlib import Path
from typing import List

import boto3
import botocore
import pandas as pd
import io
import ast

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class S3Bucket:
    """
    A class for interacting with an Amazon S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.

    Attributes:
        bucket_name (str): The name of the S3 bucket.
        s3 (boto3.client): The S3 client for low-level operations.
        s3_resource (boto3.resource): The S3 resource for high-level operations.
    """

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.s3 = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')
        if not self.bucket_exists():
            raise ValueError(f"The specified bucket '{bucket_name}' does not exist.")

    def bucket_exists(self):
        """
        Check if the specified S3 bucket exists.

        Returns:
            bool: True if the bucket exists, False otherwise.
        """
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                logging.error(f"Error checking bucket existence for '{self.bucket_name}':  {str(e)}")

    def list(self) -> List[dict]:
        """
        List all objects in the S3 bucket.

        Returns:
            list: A list of dictionaries representing objects in the bucket.

        Example:
            >>> my_bucket = S3Bucket('test-debug-nm')
            >>> objects = my_bucket.list()
        """
        objects = []
        try:
            continuation_token = None
            while True:
                if continuation_token:
                    response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix="",
                                                       ContinuationToken=continuation_token)
                else:
                    response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix="", MaxKeys=1000)
                objects.extend(response.get('Contents', []))
                if 'NextContinuationToken' in response:
                    continuation_token = response['NextContinuationToken']
                    print(continuation_token)
                else:
                    break
            return objects
        except botocore.exceptions.ClientError as e:
            logging.error(f"Error listing objects in {self.bucket_name}: {str(e)}")
            return []

    def list_directory(self, remote_directory: Path) -> List[dict]:
        """
        List objects in a specific remote directory within the S3 bucket.

        Args:
            remote_directory (Path): The path to the remote directory within the bucket.

        Returns:
            List[dict]: A list of dictionaries representing objects in the specified directory.

        Example:
            >>> my_bucket = S3Bucket('test-debug-nm')
            >>> objects_in_directory = my_bucket.list_directory(Path('path/to/directory'))
        """
        objects = []
        try:
            # Construct the prefix based on the remote directory
            prefix = str(remote_directory) + '/' if not str(remote_directory).endswith('/') else str(remote_directory)

            continuation_token = None
            while True:
                if continuation_token:
                    response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix,
                                                       ContinuationToken=continuation_token)
                else:
                    response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix, MaxKeys=1000)
                objects.extend(response.get('Contents', []))
                if 'NextContinuationToken' in response:
                    continuation_token = response['NextContinuationToken']
                else:
                    break
            return objects
        except botocore.exceptions.ClientError as e:
            logging.error(f"Error listing objects in {self.bucket_name}/{remote_directory}: {str(e)}")
            return []

    def upload_file(self, local_filepath, remote_filepath) -> bool:
        """
        Upload a file to the S3 bucket.

        Args:
            local_filepath (str): The local path to the file to upload.
            remote_filepath (str): The path of the file in the S3 bucket.

        Returns:
            bool: True if the upload was successful, False otherwise.

        Example:
            >>> my_bucket = S3Bucket('test-debug-nm')
            >>> my_bucket.upload_file('local_file.txt', 'remote_file.txt')
        """
        try:
            self.s3.upload_file(local_filepath, self.bucket_name, remote_filepath)
            logging.info(f"File '{local_filepath}' uploaded as '{remote_filepath}' to {self.bucket_name}")
            return True
        except botocore.exceptions.ClientError as e:
            logging.error(f"Error uploading file '{local_filepath}' to {self.bucket_name}:  {str(e)}")
            return False

    def upload_dataframe(self, df: pd.DataFrame, remote_filename: str) -> bool:
        """
        Upload a Pandas DataFrame to an S3 bucket.

        Args:
            df (pd.DataFrame): The DataFrame to upload.
            remote_filename (str): The remote filename within the S3 bucket.

        Returns:
            bool: True if the upload was successful, False otherwise.

        Example:
            >>> dataframe = pd.DataFrame()
            >>> my_bucket = S3Bucket('test-debug-nm')
            >>> my_bucket.upload_dataframe(dataframe, 'remote_file.txt')
        """
        try:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)  # Avoid writing the DataFrame index to the CSV
            self.s3.put_object(Body=csv_buffer.getvalue(), Bucket=self.bucket_name, Key=remote_filename)
            logging.info(
                f"DataFrame with columns '{', '.join(df.columns)}' "
                f"uploaded as '{remote_filename}' to {self.bucket_name}")
            return True
        except botocore.exceptions.ClientError as e:
            logging.error(
                f"Error uploading DataFrame with columns '{', '.join(df.columns)}' to {self.bucket_name}:  {str(e)}")
            return False

    def upload(self, text: str, remote_filename: str) -> bool:
        """
        Upload a string to an S3 bucket.

        Args:
            text (str): The string to upload.
            remote_filename (str): The remote filename within the S3 bucket.

        Returns:
            bool: True if the upload was successful, False otherwise.
        """
        try:
            self.s3.put_object(Body=text, Bucket=self.bucket_name, Key=remote_filename)
            logging.info(f"Text uploaded as '{remote_filename}' to {self.bucket_name}")
            return True
        except botocore.exceptions.ClientError as e:
            logging.error(f"Error uploading text to {self.bucket_name}:  {str(e)}")
            return False

    def get_dataframe(self, remote_filepath: str) -> pd.DataFrame:
        """
        Args:
            remote_filepath: The remote filepath in the S3 bucket the .csv file is located

        Returns:
            pd.DataFrame: The resulting dataframe

        Examples
            >>> my_bucket = S3Bucket('test-debug-nm')
            >>> my_bucket.get_dataframe('remote/path/to/csv')
        """
        try:
            raw_data = self.s3.get_object(Bucket=self.bucket_name, Key=remote_filepath)
            df = pd.read_csv(io.BytesIO(raw_data['Body'].read()))
            for column in df.columns:
                try:
                    df[column] = df[column].apply(ast.literal_eval)
                except (ValueError, SyntaxError):
                    pass  # Ignore columns that can't be converted
            return df
        except botocore.exceptions.ClientError as e:
            logging.error(f"Error getting DataFrame from {self.bucket_name}:  {str(e)}")
            return pd.DataFrame()

    def download(self, remote_file_name, local_file_path) -> bool:
        """
        Download a file from the S3 bucket.

        Args:
            remote_file_name (str): The name of the file in the S3 bucket.
            local_file_path (str): The local path where the file will be saved.

        Example:
            >>> my_bucket = S3Bucket('test-debug-nm')
            >>> my_bucket.download('remote_file.txt', 'local_copy.txt')
        """
        try:
            self.s3.download_file(self.bucket_name, remote_file_name, local_file_path)
            logging.info(f"File '{remote_file_name}' downloaded to '{local_file_path}'")
            return True
        except botocore.exceptions.ClientError as e:
            logging.error(f"Error downloading file from {self.bucket_name}: {e}")
            return False

    def delete(self, file_name: str) -> bool:
        """
        Delete a file from the S3 bucket.

        Args:
            file_name (str): The name of the file to delete from the S3 bucket.

        Example:
            >>> my_bucket = S3Bucket('test-debug-nm')
            >>> my_bucket.delete('file_to_delete.txt')
        """
        try:
            self.s3.delete_object(Bucket=self.bucket_name, Key=file_name)
            logging.info(f"File '{file_name}' deleted from {self.bucket_name}")
            return True
        except botocore.exceptions.ClientError as e:
            logging.error(f"Error deleting file '{file_name}' from {self.bucket_name}: {e}")
            return False

    def list_csv_files(self, s3_filepath: str) -> List[str]:
        """
        List all .csv files within the specified S3 filepath.

        Args:
            s3_filepath (str): The S3 filepath to search for .csv files.

        Returns:
            list: A list of .csv file names within the specified S3 filepath.
        """
        try:
            objects = self.list()
            csv_files = \
                [obj['Key'] for obj in objects if obj['Key'].startswith(s3_filepath) and obj['Key'].endswith('.csv')]
            return csv_files
        except Exception as e:
            logging.error(f"Error listing .csv files in {self.bucket_name}: {e}")
            return []
