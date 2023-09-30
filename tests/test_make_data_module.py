import pytest
from datetime import date
from src.dataloaders.dataloader import MarketDataModule  # Replace 'your_module' with the actual module name

# Sample data root and date range for testing
data_root = "/home/nikhil/PycharmProjects/TraderAI/MarketAI/data/EURUSD"
dataset_dates = (date(2022, 1, 1), date(2022, 1, 10))


@pytest.fixture
def market_data_module():
    return MarketDataModule(data_root, dataset_dates)


def test_market_data_module_setup(market_data_module):
    # Test setup method
    market_data_module.setup()
    assert market_data_module.training_dataset is not None
    assert market_data_module.validation_dataset is not None
    assert market_data_module.test_dataset is not None


def test_market_data_module_dataloaders(market_data_module):
    # Test data loader creation
    train_loader = market_data_module.train_dataloader()
    val_loader = market_data_module.val_dataloader()
    test_loader = market_data_module.test_dataloader()

    # Ensure data loaders are not None
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

# You can add more test cases as needed
