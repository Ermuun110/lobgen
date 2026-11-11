import gzip

import numpy as np
import pytest

import itch_builder as b
from lobgen.book import EMPTY_ASK, EMPTY_BID, BookError
from lobgen.itch import MSG_LEN, filter_file, iter_messages, header
from lobgen.lobster import convert, format_time, write_lobster
from lobgen.validate import check_book_transitions, check_messages

NS = 1_000_000_000
OPEN = 34_200 * NS
G, Z = 5, 6  # locate codes: GOOG, ZZZZ (unwanted)


def test_builder_lengths_match_spec():
    built = {
        b"S": b.S(0), b"R": b.R(1, "GOOG"), b"A": b.A(1, 0, 1, "B", 1, 1),
        b"F": b.F(1, 0, 1, "B", 1, 1), b"E": b.E(1, 0, 1, 1), b"C": b.C(1, 0, 1, 1, 1),
        b"X": b.X(1, 0, 1, 1), b"D": b.D(1, 0, 1), b"U": b.U(1, 0, 1, 2, 1, 1),
        b"P": b.P(1, 0, 1, 1), b"Q": b.Q(1, 0, 1, 1), b"H": b.H(1, 0, "T"),
        b"I": b.I_noii(1, 0),
    }
    for t, m in built.items():
        assert len(m) - 2 == MSG_LEN[t], t


@pytest.mark.parametrize("chunk", [52, 53, 64, 101, 1 << 20])  # 52 = largest message here
def test_filter_keeps_only_wanted_ticker_across_chunk_boundaries(tmp_path, chunk):
    stream = b"".join([
        b.S(1, "O"), b.R(G, "GOOG"), b.R(Z, "ZZZZ"),
        b.A(G, 10, 1, "B", 100, 1_000_000),
        b.A(Z, 11, 2, "S", 5, 2_000_000, sym="ZZZZ"),
        b.I_noii(Z, 12),
        b.E(G, 13, 1, 40),
        b.I_noii(G, 14),
        b.S(15, "C"),
    ])
    raw = tmp_path / "day.gz"
    raw.write_bytes(gzip.compress(stream))
    out = tmp_path / "f.itch"
    rep = filter_file(raw, ["GOOG"], out, chunk_bytes=chunk)
    kept = [header(m)[:2] for m in iter_messages(out)]
    assert kept == [("S", 0), ("R", G), ("A", G), ("E", G), ("I", G), ("S", 0)]
    assert rep["messages_seen"] == 9
    assert rep["locates"] == {"GOOG": G}


def test_filter_rejects_unknown_ticker(tmp_path):
    raw = tmp_path / "day.gz"
    raw.write_bytes(gzip.compress(b.S(1) + b.R(G, "GOOG")))
    with pytest.raises(ValueError, match="NOPE"):
        filter_file(raw, ["NOPE"], tmp_path / "f.itch")


def _day(tmp_path, msgs):
    p = tmp_path / "f.itch"
    p.write_bytes(b"".join([b.S(0, "O"), b.R(G, "GOOG")] + msgs))
    return p


def test_convert_event_mapping_and_book(tmp_path):
    t = OPEN
    path = _day(tmp_path, [
        b.A(G, OPEN - 5, 1, "B", 100, 1_000_000),     # pre-open: in book, not written
        b.A(G, t + 1, 2, "S", 50, 1_010_000),          # 1 submit ask
        b.F(G, t + 2, 3, "B", 20, 1_000_000),          # 1 submit bid (same level as ref 1)
        b.E(G, t + 3, 1, 30),                          # 4 exec bid ref 1
        b.X(G, t + 4, 2, 10),                          # 2 partial cancel ask
        b.U(G, t + 5, 3, 4, 25, 1_000_100),            # 3 delete ref 3 + 1 submit ref 4
        b.C(G, t + 6, 4, 5, 1_000_050, printable="N"), # 4 exec at limit price
        b.P(G, t + 7, 7, 1_005_000),                   # 5 hidden
        b.H(G, t + 8, "H"),                            # 7 halt
        b.H(G, t + 9, "T"),                            # 7 resume
        b.D(G, t + 10, 2),                             # 3 delete ask
        b.Q(G, t + 11, 500, 1_002_000),                # 6 cross
        b.A(G, 57_600 * NS, 9, "S", 1, 1_020_000),     # at close: excluded
    ])
    msgs, book, rep = convert(path, "GOOG", "2025-12-12", levels=2)
    assert msgs[:, 1].tolist() == [1, 1, 4, 2, 3, 1, 4, 5, 7, 7, 3, 6]
    assert msgs[:, 3].tolist() == [50, 20, 30, 10, 20, 25, 5, 7, 0, 0, 40, 500]
    assert msgs[:, 5].tolist() == [-1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1]
    assert msgs[6, 4] == 1_000_100            # 'C' recorded at the resting limit price
    assert msgs[8, 4] == -1 and msgs[9, 4] == 1
    # After the first written event the pre-open bid is visible.
    assert book[0].tolist() == [1_010_000, 50, 1_000_000, 100, EMPTY_ASK, 0, EMPTY_BID, 0]
    # Final book: ask side empty, bids 20@1_000_100 and 70@1_000_000.
    assert book[-1].tolist() == [EMPTY_ASK, 0, 1_000_100, 20, EMPTY_ASK, 0, 1_000_000, 70]
    assert rep.exec_with_price == 1 and rep.exec_with_price_nonprintable == 1
    assert rep.exec_with_price_differs == 1
    assert rep.open_orders_at_end == 3  # refs 1, 4 and the after-close ref 9
    assert check_book_transitions(msgs, book, 2)["violations"] == 0
    assert not any(check_messages(msgs).values())


def test_convert_strict_on_unknown_order(tmp_path):
    path = _day(tmp_path, [b.E(G, OPEN + 1, 99, 10)])
    with pytest.raises(BookError, match="unknown order ref 99"):
        convert(path, "GOOG", "2025-12-12")


def test_convert_strict_on_overfill(tmp_path):
    path = _day(tmp_path, [b.A(G, OPEN, 1, "B", 10, 1_000_000), b.E(G, OPEN + 1, 1, 11)])
    with pytest.raises(BookError, match="reduce 11"):
        convert(path, "GOOG", "2025-12-12")


def test_transition_check_catches_corruption(tmp_path):
    path = _day(tmp_path, [
        b.A(G, OPEN + 1, 1, "B", 100, 1_000_000),
        b.A(G, OPEN + 2, 2, "B", 50, 1_000_000),
        b.X(G, OPEN + 3, 1, 10),
    ])
    msgs, book, _ = convert(path, "GOOG", "2025-12-12", levels=2)
    assert check_book_transitions(msgs, book, 2)["violations"] == 0
    book = book.copy()
    book[2, 3] += 1  # wrong bid size after the cancel
    assert check_book_transitions(msgs, book, 2)["violations"] == 1


def test_transition_check_handles_levels_leaving_view(tmp_path):
    # 3 bid levels with a 2-level view: deleting the best bid pulls level 3 into view.
    path = _day(tmp_path, [
        b.A(G, OPEN + 1, 1, "B", 10, 1_000_300),
        b.A(G, OPEN + 2, 2, "B", 10, 1_000_200),
        b.A(G, OPEN + 3, 3, "B", 10, 1_000_100),
        b.D(G, OPEN + 4, 1),
    ])
    msgs, book, _ = convert(path, "GOOG", "2025-12-12", levels=2)
    assert check_book_transitions(msgs, book, 2)["violations"] == 0


def test_format_time_exact():
    ts = np.array([34_200 * NS, 34_200 * NS + 1, 57_599 * NS + 999_999_999])
    assert format_time(ts).tolist() == ["34200.000000000", "34200.000000001", "57599.999999999"]


def test_write_lobster_names_and_roundtrip(tmp_path):
    path = _day(tmp_path, [b.A(G, OPEN + 1, 1, "B", 100, 1_000_000)])
    msgs, book, _ = convert(path, "GOOG", "2025-12-12", levels=1)
    z, m, o = write_lobster(msgs, book, tmp_path / "out", "GOOG", "2025-12-12", 1)
    assert write_lobster(msgs, book, tmp_path / "npz_only", "GOOG", "2025-12-12", 1,
                         csv=False) == [tmp_path / "npz_only" / z.name]
    assert m.name == "GOOG_2025-12-12_34200000_57600000_message_1.csv"
    assert o.name == "GOOG_2025-12-12_34200000_57600000_orderbook_1.csv"
    assert m.read_text() == "34200.000000001,1,1,100,1000000,1\n"
    assert o.read_text() == "9999999999,0,1000000,100\n"
    assert np.load(z)["messages"].shape == (1, 6)
