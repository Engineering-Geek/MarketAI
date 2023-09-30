from utils.S3 import S3Bucket
from pathlib import Path


BUCKET_NAME = 'market-news-nm'
bucket = S3Bucket(BUCKET_NAME)
# print(bucket.get_dataframe("EURUSD/2023/8/17/news.csv")['publish_date'])
print(bucket.list_directory(Path("EURUSD/2023/3")))
