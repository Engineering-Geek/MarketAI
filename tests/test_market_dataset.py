import pytest
from datetime import date
from pathlib import Path

import torch
from src.dataloaders.dataloader import MarketDataset  # Replace 'your_module' with the actual module name

# Sample data root and date range for testing
data_root = Path("/home/nikhil/PycharmProjects/TraderAI/MarketAI/data/EURUSD")
start_date = date(2022, 1, 1)
end_date = date(2022, 1, 5)


@pytest.fixture
def market_dataset():
    return MarketDataset(data_root, start_date, end_date)


def test_market_dataset_creation(market_dataset):
    assert len(market_dataset) > 0


def test_market_dataset_getitem(market_dataset):
    # Test __getitem__ method
    item = market_dataset[0]
    assert isinstance(item, tuple)
    assert len(item) == 3
    assert isinstance(item[0], torch.Tensor)
    assert isinstance(item[1], list)
    assert isinstance(item[2], bool)

# You can add more test cases as needed
