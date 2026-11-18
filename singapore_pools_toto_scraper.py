#!/usr/bin/env python3
"""Download Singapore Pools TOTO draw results and prize breakdowns.

Uses Singapore Pools' public results pages only. It walks backwards through draw
numbers from the latest draw exposed by the official top-draws page. Some very
old draws may no longer be served by the site; missing records are logged.

Run:  python -m pip install requests beautifulsoup4 pandas openpyxl lxml
      python singapore_pools_toto_scraper.py
"""
from __future__ import annotations
import base64, re, time, logging
from pathlib import Path
from datetime import datetime
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE = 'https://www.singaporepools.com.sg'
LATEST_URL = BASE + '/DataFileArchive/Lottery/Output/toto_result_top_draws_en.html'
RESULT_URL = BASE + '/en/product/sr/Pages/toto_results.aspx'
OUT = Path(__file__).resolve().parent
DELAY_SECONDS = 0.20
MAX_CONSECUTIVE_MISSES = 12

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
s = requests.Session()
s.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; historical-results-research/1.0)'})

def get(url, **kwargs):
    for attempt in range(4):
        try:
            r = s.get(url, timeout=30, **kwargs)
            r.raise_for_status()
            return r.text
        except requests.RequestException:
            if attempt == 3: raise
            time.sleep(1.5 * (attempt + 1))

def parse_draw(html: str, expected: int | None = None):
    soup = BeautifulSoup(html, 'html.parser')
    text = re.sub(r'\s+', ' ', soup.get_text(' ', strip=True))
    m = re.search(r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})\s*\|?\s*Draw No\.?\s*(\d+)', text, re.I)
    if not m:
        m = re.search(r'(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4}).{0,80}?Draw No\.?\s*(\d+)', text, re.I)
        if not m: return None
        date_s = f'{m.group(1)} {m.group(2)} {m.group(3)}'; draw_no = int(m.group(4))
    else:
        date_s = f'{m.group(2)} {m.group(3)} {m.group(4)}'; draw_no = int(m.group(5))
    if expected is not None and draw_no != expected: return None
    try: draw_date = datetime.strptime(date_s, '%d %b %Y').date().isoformat()
    except ValueError: draw_date = date_s

    # The first seven integers following the Winning Numbers label are 6 mains + additional.
    wm = re.search(r'Winning Numbers\s*((?:\d{1,2}\s+){5}\d{1,2})\s+Additional Number\s*(\d{1,2})', text, re.I)
    if wm:
        nums = [int(x) for x in re.findall(r'\d{1,2}', wm.group(1))]
        additional = int(wm.group(2))
    else:
        return None
    if len(nums) != 6: return None

    row = {'draw_date': draw_date, 'draw_number': draw_no}
    for i, n in enumerate(nums, 1): row[f'winning_number_{i}'] = n
    row['additional_number'] = additional
    gp = re.search(r'Group 1 Prize\s*\$?([\d,]+)', text, re.I)
    row['group_1_jackpot_total_sgd'] = int(gp.group(1).replace(',', '')) if gp else None

    # Read the official Winning Shares table; retain amount per share and winning shares.
    tables = pd.read_html(html)
    share_table = None
    for t in tables:
        cols = ' '.join(str(c) for c in t.columns).lower()
        body = ' '.join(str(x) for x in t.astype(str).values.flatten()).lower()
        if 'share amount' in cols or ('group 1' in body and 'group 7' in body and 'share amount' in body):
            share_table = t; break
    for g in range(1, 8):
        row[f'group_{g}_share_amount_sgd'] = None
        row[f'group_{g}_winning_shares'] = None
    if share_table is not None:
        # Flatten rows and locate Group 1..7; handle both header and no-header layouts.
        for _, rr in share_table.iterrows():
            vals = [str(v).strip() for v in rr.tolist()]
            joined = ' '.join(vals)
            gm = re.search(r'Group\s*([1-7])', joined, re.I)
            if not gm: continue
            g = int(gm.group(1))
            money = re.search(r'\$\s*([\d,]+)', joined)
            # Winning-share count is the last integer-like cell, excluding group label and money.
            ints = [int(x.replace(',', '')) for x in re.findall(r'(?<![\w])\d[\d,]*(?![\w])', joined)]
            row[f'group_{g}_share_amount_sgd'] = int(money.group(1).replace(',', '')) if money else None
            if len(ints) >= 2: row[f'group_{g}_winning_shares'] = ints[-1]
    return row

def main():
    logging.info('Fetching latest draw number from Singapore Pools')
    latest_html = get(LATEST_URL)
    found = [int(x) for x in re.findall(r'Draw No\.?\s*(\d+)', BeautifulSoup(latest_html, 'html.parser').get_text(' ', strip=True), re.I)]
    if not found: raise RuntimeError('Could not identify latest draw number on official Singapore Pools page.')
    latest = max(found)
    logging.info('Latest draw detected: %s', latest)
    records, misses = [], 0
    for draw in range(latest, 0, -1):
        token = base64.b64encode(f'DrawNumber={draw}'.encode()).decode()
        url = RESULT_URL + '?sppl=' + requests.utils.quote(token, safe='')
        try:
            html = get(url)
            record = parse_draw(html, draw)
            if record:
                records.append(record); misses = 0
                logging.info('Captured draw %s (%s)', draw, record['draw_date'])
            else:
                misses += 1; logging.info('No parseable official result for draw %s', draw)
        except Exception as e:
            misses += 1; logging.warning('Draw %s failed: %s', draw, e)
        if misses >= MAX_CONSECUTIVE_MISSES:
            logging.info('Stopping after %s consecutive unavailable draws.', misses)
            break
        time.sleep(DELAY_SECONDS)
    if not records: raise RuntimeError('No draw records were parsed. Check whether Singapore Pools changed its page structure.')
    df = pd.DataFrame(records).sort_values(['draw_date', 'draw_number'], na_position='last')
    # Keep numbers as integers; CSV and Excel outputs share the same columns.
    csv_path = OUT / 'singapore_pools_toto_historical.csv'
    xlsx_path = OUT / 'singapore_pools_toto_historical.xlsx'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='TOTO Draws')
        ws = writer.sheets['TOTO Draws']; ws.freeze_panes = 'A2'; ws.auto_filter.ref = ws.dimensions
        for col in ws.columns:
            letter = col[0].column_letter
            width = min(max(len(str(c.value or '')) for c in col) + 2, 28)
            ws.column_dimensions[letter].width = width
        meta = pd.DataFrame([
            ['Source', 'Singapore Pools official TOTO results pages'],
            ['Official latest-draw archive', LATEST_URL],
            ['Individual draw URL pattern', RESULT_URL + '?sppl=base64(DrawNumber=N)'],
            ['Records captured', len(df)],
            ['First date in output', str(df['draw_date'].min())],
            ['Last date in output', str(df['draw_date'].max())],
            ['Note', 'Unavailable/legacy draws may be absent. Blank prize fields mean not parsed/published on the source page.'],
        ], columns=['Field', 'Value'])
        meta.to_excel(writer, index=False, sheet_name='README')
    logging.info('Saved %s rows to %s and %s', len(df), csv_path, xlsx_path)

if __name__ == '__main__': main()
