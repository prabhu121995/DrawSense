#!/usr/bin/env python3
"""
Singapore Pools 4D historical results scraper.

How the official site serves 4D (same idea as TOTO, see toto_scraper.py):
  * Every draw is listed in a static file:
        /DataFileArchive/Lottery/Output/fourd_result_draw_list_en.html
    Each <option value="NNNN" queryString="sppl=<base64('DrawNumber=NNNN')>" isCancelled="">.
  * One draw's result is rendered by the server at:
        /en/product/pages/4d_results.aspx?sppl=<base64>
    inside <div class="divSingleDraw">: drawDate, drawNumber, tdFirstPrize / tdSecondPrize / tdThirdPrize,
    tbodyStarterPrizes (10 numbers) and tbodyConsolationPrizes (10 numbers).

Output CSV columns (numbers are kept as 4-character text, so 0018 stays 0018):
  draw_number, draw_date, first, second, third, starter_1..starter_10, consolation_1..consolation_10

Usage:
  python fourd_scraper.py                 # only new draws since the last one in fourd_results.csv
  python fourd_scraper.py --full          # the whole official list (fills gaps)
  python fourd_scraper.py --latest 20
  python fourd_scraper.py --debug 5566    # save raw HTML of one draw and print the parsed row
"""

import argparse
import csv
import os
import random
import re
import sys
import time

from bs4 import BeautifulSoup

# Same folder: reuse the HTTP session, retries and draw-list parsing of the TOTO scraper.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toto_scraper import BASE, fetch, make_session, sppl_for  # noqa: E402

DRAW_LIST_URL = BASE + "/DataFileArchive/Lottery/Output/fourd_result_draw_list_en.html"
DRAW_URL = BASE + "/en/product/pages/4d_results.aspx?sppl={sppl}"
REFERER = BASE + "/en/product/pages/4d_results.aspx"

N_STARTER = N_CONSOLATION = 10
FIELDS = (["draw_number", "draw_date", "first", "second", "third"]
          + [f"starter_{i}" for i in range(1, N_STARTER + 1)]
          + [f"consolation_{i}" for i in range(1, N_CONSOLATION + 1)])


def get_draw_list(session):
    """Return [(draw_number, sppl)], newest first; cancelled draws are left out."""
    html = fetch(session, DRAW_LIST_URL)
    if not html:
        return []
    out, seen = [], set()
    for opt in BeautifulSoup(html, "html.parser").find_all("option"):
        v = (opt.get("value") or "").strip()
        if not v.isdigit() or (opt.get("iscancelled") or "").strip().lower() in ("true", "1", "y", "yes"):
            continue
        num = int(v)
        m = re.search(r"sppl=([A-Za-z0-9+/=]+)", opt.get("querystring") or "")
        if num not in seen:
            seen.add(num)
            out.append((num, m.group(1) if m else sppl_for(num)))
    return out


def _four(text):
    t = re.sub(r"\D", "", text or "")
    return t if len(t) == 4 else ""


def parse_draw(html, expected_draw=None):
    """Parse one 4D draw page. Returns a dict with FIELDS, or None if the result is not on the page."""
    soup = BeautifulSoup(html, "html.parser")
    block = soup.find(class_="divSingleDraw") or soup.find(class_="tables-wrap") or soup
    row = {k: "" for k in FIELDS}

    dn = block.find(class_="drawNumber")
    m = re.search(r"(\d{3,5})", dn.get_text(" ", strip=True) if dn else "")
    row["draw_number"] = m.group(1) if m else (str(expected_draw) if expected_draw else "")
    dd = block.find(class_="drawDate")
    row["draw_date"] = dd.get_text(" ", strip=True) if dd else ""

    for key, cls in (("first", "tdFirstPrize"), ("second", "tdSecondPrize"), ("third", "tdThirdPrize")):
        el = block.find(class_=cls)
        row[key] = _four(el.get_text(strip=True) if el else "")

    for key, cls, n in (("starter", "tbodyStarterPrizes", N_STARTER),
                        ("consolation", "tbodyConsolationPrizes", N_CONSOLATION)):
        body = block.find(class_=cls)
        nums = [_four(td.get_text(strip=True)) for td in body.find_all("td")] if body else []
        nums = [x for x in nums if x][:n]
        for i, x in enumerate(nums, 1):
            row[f"{key}_{i}"] = x

    if not (row["first"] and row["second"] and row["third"] and row["draw_date"]):
        return None
    return row


# --------------------------------------------------------------------------- CSV helpers
def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if (r.get("draw_number") or "").isdigit()]


def write_rows(path, rows):
    rows = sorted(rows, key=lambda r: int(r["draw_number"]))
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)  # atomic: never leaves a half-written file


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Scrape Singapore Pools 4D results")
    ap.add_argument("-o", "--output", default="fourd_results.csv")
    ap.add_argument("--latest", type=int, help="only the N most recent draws")
    ap.add_argument("--full", action="store_true", help="check the whole draw list and fill gaps")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--debug", type=int, metavar="DRAW", help="fetch one draw, save raw HTML, print the parsed row")
    args = ap.parse_args()

    session = make_session()
    session.headers["Referer"] = REFERER

    if args.debug:
        html = fetch(session, DRAW_URL.format(sppl=sppl_for(args.debug)))
        if not html:
            sys.exit("Could not fetch page.")
        fn = f"debug_4d_{args.debug}.html"
        with open(fn, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Saved raw HTML to {fn}")
        print(parse_draw(html, args.debug))
        return

    rows = read_rows(args.output)
    done = {r["draw_number"] for r in rows}
    last_num = max((int(n) for n in done), default=None)
    incremental = last_num is not None and not args.full and not args.latest
    if last_num is not None:
        print(f"4D: last draw in {args.output}: #{last_num}")

    print("4D: fetching draw list ...")
    draws = get_draw_list(session)
    if not draws:
        sys.exit("4D: draw list unavailable - try again later.")
    print(f"4D: {len(draws)} draws in official list (#{min(d[0] for d in draws)} - #{max(d[0] for d in draws)})")
    if incremental:
        draws = [d for d in draws if d[0] > last_num]
    if args.latest:
        draws = sorted(draws, reverse=True)[: args.latest]
    todo = sorted(d for d in draws if str(d[0]) not in done)  # oldest first: a stop mid-run leaves no gap
    if not todo:
        print("4D: already up to date - no new draws to fetch.")
        return
    print(f"4D: {len(todo)} draws to fetch ({len(done)} already in {args.output})")

    ok = failed = 0
    for i, (num, sppl) in enumerate(todo, 1):
        html = fetch(session, DRAW_URL.format(sppl=sppl))
        row = parse_draw(html, num) if html else None
        if row and row["draw_number"] == str(num):
            append_row(args.output, row)
            ok += 1
            print(f"4D [{i}/{len(todo)}] #{num} {row['draw_date']}: {row['first']} {row['second']} {row['third']}")
        else:
            failed += 1
            print(f"4D [{i}/{len(todo)}] #{num}: no parseable result - skipped")
        time.sleep(args.delay + random.random() * 0.5)

    write_rows(args.output, read_rows(args.output))  # keep the file sorted by draw number
    print(f"\n4D done. {ok} saved, {failed} skipped. Output: {args.output}")


if __name__ == "__main__":
    main()
