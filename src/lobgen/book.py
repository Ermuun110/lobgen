"""Order-level limit order book. Prices are integer dollars x 10000 (ITCH == LOBSTER units)."""

from __future__ import annotations

from sortedcontainers import SortedDict

BUY, SELL = 1, -1
EMPTY_ASK = 9_999_999_999
EMPTY_BID = -9_999_999_999


class BookError(RuntimeError):
    pass


class OrderBook:
    def __init__(self) -> None:
        self.orders: dict[int, list[int]] = {}  # ref -> [side, price, shares]
        self.bids: SortedDict = SortedDict()     # price -> total shares
        self.asks: SortedDict = SortedDict()

    def _levels(self, side: int) -> SortedDict:
        return self.bids if side == BUY else self.asks

    def add(self, ref: int, side: int, price: int, shares: int) -> None:
        if ref in self.orders:
            raise BookError(f"duplicate order ref {ref}")
        if shares <= 0:
            raise BookError(f"non-positive add size {shares} for ref {ref}")
        self.orders[ref] = [side, price, shares]
        lv = self._levels(side)
        lv[price] = lv.get(price, 0) + shares

    def reduce(self, ref: int, shares: int) -> tuple[int, int, int]:
        """Remove `shares` from an order. Returns (side, price, remaining)."""
        o = self.orders.get(ref)
        if o is None:
            raise BookError(f"unknown order ref {ref}")
        side, price, have = o
        if shares <= 0 or shares > have:
            raise BookError(f"reduce {shares} from ref {ref} holding {have}")
        remaining = have - shares
        lv = self._levels(side)
        left = lv[price] - shares
        if left:
            lv[price] = left
        else:
            del lv[price]
        if remaining:
            o[2] = remaining
        else:
            del self.orders[ref]
        return side, price, remaining

    def delete(self, ref: int) -> tuple[int, int, int]:
        """Remove an order entirely. Returns (side, price, shares removed)."""
        o = self.orders.get(ref)
        if o is None:
            raise BookError(f"unknown order ref {ref}")
        side, price, shares = o
        self.reduce(ref, shares)
        return side, price, shares

    def best_bid(self) -> int | None:
        return self.bids.peekitem(-1)[0] if self.bids else None

    def best_ask(self) -> int | None:
        return self.asks.peekitem(0)[0] if self.asks else None

    def snapshot(self, levels: int) -> list[int]:
        """LOBSTER orderbook row: askP1, askV1, bidP1, bidV1, askP2, ... (4 * levels ints)."""
        row: list[int] = []
        ak, av = self.asks.keys(), self.asks
        bk, bv = self.bids.keys(), self.bids
        na, nb = len(ak), len(bk)
        for i in range(levels):
            if i < na:
                p = ak[i]
                row += (p, av[p])
            else:
                row += (EMPTY_ASK, 0)
            if i < nb:
                p = bk[nb - 1 - i]
                row += (p, bv[p])
            else:
                row += (EMPTY_BID, 0)
        return row
