"""Axis & Mirror metrics, the 4D page parser and the shared SQLite database."""
import sys
from pathlib import Path

import pytest

from toto_ml import db
from toto_ml.metrics import metrics, reference_chart

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fourd_scraper  # noqa: E402

# rows copied from the "Axis & Mirror Numbers Reference Chart (01 - 89)": number, root, axis, mirror
DOCUMENT_ROWS = [("01", 1, "56", "10"), ("10", 1, "65", "01"), ("14", 5, "69", "41"), ("27", 9, "72", "72"),
                 ("30", 3, "85", "03"), ("46", 1, "91", "64"), ("50", 5, "05", "05"), ("55", 1, "00", "55"),
                 ("60", 6, "15", "06"), ("89", 8, "34", "98")]


@pytest.mark.parametrize("number,root,axis,mirror", DOCUMENT_ROWS)
def test_metrics_match_document_chart(number, root, axis, mirror):
    m = metrics(int(number))
    assert (m["number"], m["root"], m["axis"], m["mirror"]) == (number, root, axis, mirror)


def test_four_digit_metrics_and_chart():
    m = metrics("725", 4)
    assert m == {"number": "0725", "digit_sum": 14, "root": 5, "axis": "5270", "mirror": "5270",
                 "front": "07", "back": "25"}
    assert metrics("0000", 4)["root"] == 0
    chart = reference_chart()
    assert len(chart) == 100 and sum(r["in_chart"] for r in chart) == 89


PAGE = """<div class="divSingleDraw"><div class="tables-wrap"><table><thead><tr>
<th class="drawDate">Wed, 25 Nov 2026</th><th class="drawNumber">Draw No. 5566</th></tr></thead><tbody>
<tr><th>1st Prize</th><td class="tdFirstPrize">8064</td></tr><tr><th>2nd Prize</th><td class="tdSecondPrize">2112</td></tr>
<tr><th>3rd Prize</th><td class="tdThirdPrize">8962</td></tr></tbody></table>
<table><tbody class="tbodyStarterPrizes">""" + "".join(f"<tr><td>{i:04d}</td><td>{i + 10:04d}</td></tr>" for i in range(5)) + \
"""</tbody></table><table><tbody class="tbodyConsolationPrizes">""" + \
"".join(f"<tr><td>{i + 100:04d}</td><td>{i + 200:04d}</td></tr>" for i in range(5)) + "</tbody></table></div></div>"


def test_fourd_parser():
    row = fourd_scraper.parse_draw(PAGE)
    assert row["draw_number"] == "5566" and row["draw_date"] == "Wed, 25 Nov 2026"
    assert (row["first"], row["second"], row["third"]) == ("8064", "2112", "8962")
    assert row["starter_1"] == "0000" and row["starter_10"] == "0014" and row["consolation_10"] == "0204"
    assert fourd_scraper.parse_draw("<html>no result</html>") is None


def test_sync_both_games_into_one_database(tmp_path):
    toto = tmp_path / "toto.csv"
    toto.write_text('draw_number,draw_date,n1,n2,n3,n4,n5,n6,additional,group1_prize\n'
                    '4236,"Thu, 19 Nov 2026",8,26,28,37,41,42,34,1183486\n'
                    ',"Mon, 23 Nov 2026",25,27,37,43,47,49,17,\n', encoding="utf-8")   # hand-typed: no number
    fd = tmp_path / "fourd.csv"
    row = fourd_scraper.parse_draw(PAGE)
    fourd_scraper.write_rows(str(fd), [row])
    dbf = tmp_path / "x.db"
    assert db.sync_all(toto, fd, dbf) == {"toto": 2, "4d": 1}
    assert db.sync_all(toto, fd, dbf) == {"toto": 2, "4d": 1}                      # re-run: no duplicates
    counts = dict(db.table_counts(dbf).values)
    assert counts["draws"] == 3 and counts["draw_numbers"] == 2 * 7 + 23 and counts["number_metrics"] == 100
    v = db.draw_numbers("4d", ["1st"], db_path=dbf).iloc[0]
    assert (v["number"], v["front"], v["back"], v["front_axis"], v["back_mirror"]) == ("8064", "80", "64", "35", "46")
    hit = db.lookup("toto", "14", dbf)                                              # 14: mirror 41 was drawn
    assert hit["metrics"]["mirror"] == "41" and len(hit["number"]) == 0 and len(hit["mirror"]) == 1
    p = db.lookup("4d", "4608", dbf)["permutation"]                                # 8064 is a permutation
    assert p["number"].tolist() == ["8064"]
