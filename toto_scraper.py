#!/usr/bin/env python3
"""
Singapore Pools TOTO historical results scraper.

How the official site works (why naive scrapers fail):
  * toto_results.aspx itself is mostly an empty shell - results are injected
    by JavaScript, so requests.get() on the bare page returns no numbers.
  * The list of every draw lives in a static file:
        /DataFileArchive/Lottery/Output/toto_result_draw_list_en.html
    Each <option> carries querystring="sppl=<base64>", where the base64 text
    decodes to "DrawNumber=NNNN".
  * A single draw's full result is served at:
        /en/product/sr/Pages/toto_results.aspx?sppl=<base64("DrawNumber=NNNN")>

Output CSV columns:
  draw_number, draw_date, n1..n6, additional, group1_prize,
  g1_share_amount, g1_winning_shares, ... g7_share_amount, g7_winning_shares

Usage:
  pip install requests beautifulsoup4
  python toto_scraper.py                       # all draws in the official list
  python toto_scraper.py --start 3800 --end 4100
  python toto_scraper.py --latest 50
  python toto_scraper.py --debug 4000          # save raw HTML for one draw

Already-scraped draws in the output CSV are skipped, so you can stop and
re-run safely. Always compare a sample with the official site before relying
on the data.
"""

import argparse
import base64
import csv
import os
import random
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

BASE = "https://www.singaporepools.com.sg"
DRAW_LIST_URL = BASE + "/DataFileArchive/Lottery/Output/toto_result_draw_list_en.html"
DRAW_URL = BASE + "/en/product/sr/Pages/toto_results.aspx?sppl={sppl}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE + "/en/product/sr/Pages/toto_results.aspx",
}

FIELDS = (
    ["draw_number", "draw_date", "n1", "n2", "n3", "n4", "n5", "n6",
     "additional", "group1_prize"]
    + [f"g{g}_{k}" for g in range(1, 8) for k in ("share_amount", "winning_shares")]
)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def make_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def fetch(session, url, retries=4, timeout=30):
    """GET with retries and exponential backoff. Returns text or None."""
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200 and r.text.strip():
                r.encoding = r.apparent_encoding or "utf-8"
                return r.text
            print(f"  HTTP {r.status_code} for {url} (attempt {attempt})", file=sys.stderr)
        except requests.RequestException as e:
            print(f"  {type(e).__name__}: {e} (attempt {attempt})", file=sys.stderr)
        time.sleep(2 ** attempt + random.random())
    return None


def sppl_for(draw_number):
    return base64.b64encode(f"DrawNumber={draw_number}".encode()).decode()


# --------------------------------------------------------------------------- #
# Draw list
# --------------------------------------------------------------------------- #
def get_draw_list(session):
    """Return list of (draw_number:int, sppl:str), newest first."""
    html = fetch(session, DRAW_LIST_URL)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    draws = []
    for opt in soup.find_all("option"):
        qs = opt.get("querystring") or opt.get("value") or ""
        m = re.search(r"sppl=([A-Za-z0-9+/=]+)", qs)
        sppl = m.group(1) if m else None
        num = None
        if sppl:
            try:
                decoded = base64.b64decode(sppl).decode(errors="ignore")
                dm = re.search(r"DrawNumber=(\d+)", decoded)
                num = int(dm.group(1)) if dm else None
            except Exception:
                pass
        if num is None:
            v = (opt.get("value") or "").strip()
            if v.isdigit():
                num = int(v)
                sppl = sppl_for(num)
        if num is not None:
            draws.append((num, sppl))
    # de-duplicate, keep order
    seen, out = set(), []
    for d in draws:
        if d[0] not in seen:
            seen.add(d[0])
            out.append(d)
    return out


# --------------------------------------------------------------------------- #
# Parsing one draw page
# --------------------------------------------------------------------------- #
def clean_money(text):
    """'$1,234,567' -> '1234567'; '-' or blank -> ''."""
    if not text:
        return ""
    t = re.sub(r"[^\d.]", "", text)
    return t if re.search(r"\d", t) else ""


def clean_int(text):
    if not text:
        return ""
    t = re.sub(r"[^\d]", "", text)
    return t


def first_text(soup, cls):
    el = soup.find(class_=cls)
    return el.get_text(" ", strip=True) if el else ""


def parse_draw(html, expected_draw=None):
    """Parse a draw page. Returns dict or None if winning numbers not found."""
    soup = BeautifulSoup(html, "html.parser")
    row = {k: "" for k in FIELDS}

    # The page can contain several result blocks; use the first one.
    block = soup.find(class_="tables-wrap") or soup

    # Draw number / date
    dn = first_text(block, "drawNumber")
    m = re.search(r"(\d{3,5})", dn)
    if m:
        row["draw_number"] = m.group(1)
    elif expected_draw:
        row["draw_number"] = str(expected_draw)
    row["draw_date"] = first_text(block, "drawDate")

    # Winning numbers: <td class="win1"> ... <td class="win6">
    nums = [first_text(block, f"win{i}") for i in range(1, 7)]
    if not all(n.strip().isdigit() for n in nums):
        # Fallback: first table after a "Winning Numbers" heading
        nums = []
        hdr = block.find(string=re.compile(r"Winning Numbers", re.I))
        if hdr:
            tbl = hdr.find_parent("table")
            if tbl:
                nums = [c.get_text(strip=True) for c in tbl.find_all("td")
                        if c.get_text(strip=True).isdigit()][:6]
    if len(nums) != 6 or not all(n.isdigit() for n in nums):
        return None
    for i, n in enumerate(nums, 1):
        row[f"n{i}"] = n

    add = first_text(block, "additional")
    if not add.isdigit():
        hdr = block.find(string=re.compile(r"Additional Number", re.I))
        if hdr:
            tbl = hdr.find_parent("table")
            if tbl:
                tds = [c.get_text(strip=True) for c in tbl.find_all("td")]
                add = next((t for t in tds if t.isdigit()), "")
    row["additional"] = add

    # Group 1 prize (jackpot total)
    row["group1_prize"] = clean_money(first_text(block, "jackpotPrize"))
    if not row["group1_prize"]:
        hdr = block.find(string=re.compile(r"Group 1 Prize", re.I))
        if hdr:
            tbl = hdr.find_parent("table")
            if tbl:
                row["group1_prize"] = clean_money(tbl.get_text(" ", strip=True))

    # Winning shares table: Group | Share Amount | No. of Winning Shares
    shares_tbl = block.find("table", class_=re.compile("tableWinningShares"))
    if shares_tbl is None:
        hdr = block.find(string=re.compile(r"Winning Shares", re.I))
        shares_tbl = hdr.find_parent("table") if hdr else None
    if shares_tbl is not None:
        for tr in shares_tbl.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if len(cells) < 3:
                continue
            gm = re.search(r"Group\s*(\d)", cells[0], re.I) or re.fullmatch(r"(\d)", cells[0])
            if not gm:
                continue
            g = int(gm.group(1))
            if 1 <= g <= 7:
                row[f"g{g}_share_amount"] = clean_money(cells[1])
                row[f"g{g}_winning_shares"] = clean_int(cells[2])

    return row


# --------------------------------------------------------------------------- #
# CSV helpers
# --------------------------------------------------------------------------- #
def load_done(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {r["draw_number"] for r in csv.DictReader(f) if r.get("draw_number")}


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def sort_csv(path):
    if not os.path.exists(path):
        return
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["draw_number"] or 0))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Scrape Singapore Pools TOTO results")
    ap.add_argument("-o", "--output", default="toto_results.csv")
    ap.add_argument("--start", type=int, help="first draw number")
    ap.add_argument("--end", type=int, help="last draw number")
    ap.add_argument("--latest", type=int, help="only the N most recent draws")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between requests")
    ap.add_argument("--debug", type=int, metavar="DRAW",
                    help="fetch one draw, save raw HTML, print parsed result")
    args = ap.parse_args()

    session = make_session()

    if args.debug:
        url = DRAW_URL.format(sppl=sppl_for(args.debug))
        print("GET", url)
        html = fetch(session, url)
        if not html:
            sys.exit("Could not fetch page.")
        fn = f"debug_draw_{args.debug}.html"
        with open(fn, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Saved raw HTML to {fn} ({len(html):,} chars)")
        print(parse_draw(html, args.debug))
        return

    print("Fetching draw list ...")
    draws = get_draw_list(session)
    if draws:
        print(f"  {len(draws)} draws in official list "
              f"(#{min(d[0] for d in draws)} - #{max(d[0] for d in draws)})")
    else:
        print("  Draw list unavailable; falling back to --start/--end range.")
        if not (args.start and args.end):
            sys.exit("Provide --start and --end when the draw list can't be fetched.")
        draws = [(n, sppl_for(n)) for n in range(args.end, args.start - 1, -1)]

    if args.start:
        draws = [d for d in draws if d[0] >= args.start]
    if args.end:
        draws = [d for d in draws if d[0] <= args.end]
    draws.sort(key=lambda d: d[0], reverse=True)
    if args.latest:
        draws = draws[: args.latest]

    done = load_done(args.output)
    todo = [d for d in draws if str(d[0]) not in done]
    print(f"{len(todo)} draws to fetch ({len(done)} already in {args.output})")

    ok = failed = 0
    for i, (num, sppl) in enumerate(todo, 1):
        html = fetch(session, DRAW_URL.format(sppl=sppl))
        row = parse_draw(html, num) if html else None
        if row:
            append_row(args.output, row)
            ok += 1
            nums = " ".join(row[f"n{k}"] for k in range(1, 7))
            print(f"[{i}/{len(todo)}] #{row['draw_number']} {row['draw_date']}: "
                  f"{nums} + {row['additional']}")
        else:
            failed += 1
            print(f"[{i}/{len(todo)}] #{num}: no parseable result - skipped")
        time.sleep(args.delay + random.random() * 0.5)

    sort_csv(args.output)
    print(f"\nDone. {ok} saved, {failed} skipped. Output: {args.output}")


if __name__ == "__main__":
    main()
