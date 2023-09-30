import datetime
from typing import Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path
from datetime import date
import pandas as pd
from datetime import timedelta

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class MarketDataset(Dataset):
    """
    Dataset class for combining forex and news data.

    Args:
        data_root (Path): Root directory of the data.
        start (date): Start date for the dataset.
        end (date): End date for the dataset.
        summary (bool, optional): Use news summaries if True, otherwise use full news text. Defaults to True.
        window (int, optional): Window size. Defaults to 1.

    Attributes:
        summary (bool): Whether to use news summaries or full text.
        window (int): Window size.
        last_date (datetime.date): Last processed date.

    Examples:
        >>> dataset = MarketDataset(Path("filepath"), date(2020, 1, 1), date(2022, 1, 1))
        >>> for i in range(len(dataset)):
        >>>     news, forex, new_day = dataset[i]

    """

    def __init__(self, data_root: Path, start: date, end: date, summary: bool = True, window: int = 1):
        self.summary = summary
        self.window = window
        self.last_date: datetime.date = start - datetime.timedelta(days=1)

        days: List[datetime.datetime] = pd.date_range(start=start, end=end, freq='D').tolist()
        all_news = []
        all_rates = []
        for day in days:
            year, month, day = day.year, day.month, day.day
            path = data_root / str(year) / str(month) / str(day)
            if path.exists():
                news_df = self._get_news(path)
                rate_df = self._get_forex(path)
                if not news_df.empty:
                    all_news.append(news_df)
                if not rate_df.empty:
                    all_rates.append(rate_df)
            else:
                raise NotADirectoryError(f"Path {path} does not exist; date: {year}-{month}-{day}")

        self.news_df = pd.concat(all_news) if all_news else pd.DataFrame()
        self.forex_df = pd.concat(all_rates) if all_rates else pd.DataFrame()

    @staticmethod
    def _get_forex(path: Path):
        """
        Get forex data from a given path.

        Args:
            path (Path): Path to forex data.

        Returns:
            DataFrame: Forex data.

        """
        rate_df = pd.read_csv(path / "forex.csv").dropna().drop_duplicates()
        rate_df['time'] = pd.to_datetime(rate_df['time'], unit='s')
        rate_df = rate_df.set_index('time').resample('1Min').mean().dropna()
        rate_df['day'] = rate_df.index.date
        return rate_df

    @staticmethod
    def _get_news(path: Path):
        """
        Get news data from a given path.

        Args:
            path (Path): Path to news data.

        Returns:
            DataFrame: News data.

        """
        # Read CSV and drop duplicates
        news_df = pd.read_csv(path / "news.csv").drop_duplicates()

        # Drop rows with missing publish_date
        news_df.dropna(subset=['publish_date'], inplace=True)

        # Convert publish_date to datetime, handling errors with 'coerce'
        news_df['publish_date'] = pd.to_datetime(news_df['publish_date'], format="%Y-%m-%d %H:%M:%S", errors='coerce')

        # Remove rows with invalid datetime values
        news_df.dropna(subset=['publish_date'], inplace=True)

        # Remove timezone information and set publish_date as index
        news_df['publish_date'] = news_df['publish_date'].dt.tz_localize(None)

        # Convert datetime to date
        news_df['publish_date'] = news_df['publish_date'].dt.date

        # Group the DataFrame by 'publish_date' and aggregate other columns into lists
        news_df = news_df.groupby('publish_date').agg(lambda x: x.tolist())
        return news_df

    def _flatten_list(self, nested_list):
        """
        Flatten a nested list.

        Args:
            nested_list (list): Nested list to be flattened.

        Returns:
            list: Flattened list.

        """
        flattened_list = []
        for item in nested_list:
            if isinstance(item, list):
                flattened_list.extend(self._flatten_list(item))
            else:
                flattened_list.append(item)
        return flattened_list

    def __len__(self) -> int:
        return len(self.forex_df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, List[str], bool]:
        """
        Get data for a specific index.

        Args:
            idx (int): Index of the data.

        Returns:
            Tuple[Tensor, List[str], bool]: Tuple containing tensors for forex data, news data, and a boolean indicating
            if it's a new day.

        """
        forex_row = self.forex_df.iloc[idx]
        idx_date = forex_row['day']
        avg_price = np.mean(forex_row[['bid', 'ask']].values)
        spread = np.mean(forex_row[['bid', 'ask']].values) - np.min(forex_row[['bid', 'ask']].values)
        tensor_1 = torch.tensor([avg_price, spread], dtype=torch.float32)
        news = self.news_df.loc[idx_date]['summary'] if self.summary else self.news_df.loc[date]['text']
        news = news.tolist() if isinstance(news, pd.Series) else news
        news = self._flatten_list(news)
        new_day = idx_date == self.last_date
        self.last_date = idx_date
        return tensor_1, news, new_day


class MarketDataModule(LightningDataModule):
    """
    LightningDataModule for preparing market data for training, validation, and testing.

    Args:
        data_root (str): Root directory of the data.
        dataset_dates (Tuple[date, date]): Start and end dates for the dataset.
        batch_size (int, optional): Batch size for data loaders. Defaults to 32.
        num_workers (int, optional): Number of data loader workers. Defaults to 4.
        summary (bool, optional): Use news summaries if True, otherwise use full news text. Defaults to True.
        window (int, optional): Window size. Defaults to 1.
        train_val_split (float, optional): Fraction of data to use for training and validation split. Defaults to 0.7.
        test_split (float, optional): Fraction of data to use for testing. Defaults to 0.15.

    Examples:
        >>> data_module = MarketDataModule(data_root="data", dataset_dates=(date(2022, 1, 1), date(2022, 12, 31)))
        >>> data_module.setup()
        >>> train_loader = data_module.train_dataloader()
        >>> val_loader = data_module.val_dataloader()
        >>> test_loader = data_module.test_dataloader()
    """

    def __init__(self, data_root: str, dataset_dates: Tuple[date, date], batch_size: int = 32, num_workers: int = 4,
                 summary: bool = True, window: int = 1, train_val_split: float = 0.7, test_split: float = 0.15):
        super().__init__()
        self.data_root = data_root
        self.dataset_dates = dataset_dates
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.summary = summary
        self.window = window
        self.train_val_split = train_val_split
        self.test_split = test_split
        self.training_dataset: MarketDataset = None
        self.validation_dataset: MarketDataset = None
        self.test_dataset: MarketDataset = None

    def setup(self, stage=None):
        """
        Setup datasets for training, validation, and testing.

        Args:
            stage (str, optional): Stage of setup ('fit', 'validate', or 'test'). Defaults to None.

        Notes:
            - If stage is 'fit' or None, it sets up the training, validation, and testing datasets.
        """
        if stage == 'fit' or stage is None:
            train_end_date = self.dataset_dates[0] + timedelta(
                days=int(self.train_val_split * (self.dataset_dates[1] - self.dataset_dates[0]).days))
            self.training_dataset = MarketDataset(Path(self.data_root), self.dataset_dates[0], train_end_date,
                                                  window=self.window, summary=self.summary)
            val_end_date = train_end_date + timedelta(
                days=int(self.test_split * (self.dataset_dates[1] - self.dataset_dates[0]).days))
            self.validation_dataset = MarketDataset(Path(self.data_root), train_end_date, val_end_date,
                                                    window=self.window, summary=self.summary)
            self.test_dataset = MarketDataset(Path(self.data_root), val_end_date, self.dataset_dates[1],
                                              window=self.window, summary=self.summary)

    def train_dataloader(self):
        """
        Return the training data loader.

        Returns:
            DataLoader: Training data loader.
        """
        return DataLoader(self.training_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        """
        Return the validation data loader.

        Returns:
            DataLoader: Validation data loader.
        """
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        """
        Return the test data loader.

        Returns:
            DataLoader: Test data loader.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
