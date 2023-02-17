from dataclasses import dataclass
from datetime import datetime


@dataclass
class Order:
    """Base class for Orders."""
    timestamp: datetime
    direction: str #buy or sell
    ticker: str

    def __lt__(self, other):
        return self.timestamp < other.timestamp


@dataclass
class MarketOrder(Order):
    volume: int
    price: float


@dataclass
class OrderDict:
    timestamp: datetime
    price: float
    volume: int
    direction: str
    ticker: str