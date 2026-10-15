"""marks.load_spots: closes a user adds in data/external/closes.csv (synthetic numbers; no real data)."""
import tempfile
from datetime import date
from pathlib import Path

from _harness import check

from definedtarget import marks

with tempfile.TemporaryDirectory() as tmp:
    ext = Path(tmp) / "external"
    ext.mkdir()
    (ext / "closes.csv").write_text("date,symbol,close,source\n"
                                    "2026-01-05,SPY,700.00,my broker\n"
                                    "2026-01-05,SPX,7020.00,my broker\n"
                                    "2026-01-06,AMEX:SPY,701.50,exchange prefix\n", encoding="utf-8")
    spots = marks.load_spots(tmp)
    check("a user close for SPY loads, with the S&P close beside it", spots.get(date(2026, 1, 5))[:2] == (700.0, 7020.0),
          spots)
    check("an exchange prefix (AMEX:SPY) reads as SPY", spots.get(date(2026, 1, 6))[0] == 701.5, spots)
    check("each row records where its price came from", "closes.csv" in spots[date(2026, 1, 5)][2])
    check("a date with no SPY close has no entry", date(2026, 1, 7) not in spots)
