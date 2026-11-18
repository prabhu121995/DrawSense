Singapore Pools TOTO historical results scraper

This package creates:
  - singapore_pools_toto_historical.csv
  - singapore_pools_toto_historical.xlsx

It requests draw pages from Singapore Pools' official website and extracts:
  - draw date and draw number
  - six winning numbers and additional number
  - Group 1 jackpot total
  - Groups 1-7 share amount and winning-share count (where present and parseable)

IMPORTANT:
The current web session could verify the official endpoints but could not directly
fetch and execute thousands of historical requests. Therefore this package contains
the scraper, not a falsely represented complete historical dataset. The site may
not expose every legacy draw; unavailable or unparsable draws are omitted and
blank prize fields are retained as blank. Review the output and compare with the
official Singapore Pools results page before relying on it.

Requirements (Python 3.10+):
  python -m pip install requests beautifulsoup4 pandas openpyxl lxml

Run from CMD:
  cd path\to\toto_scraper
  python singapore_pools_toto_scraper.py

Official source:
  https://www.singaporepools.com.sg/en/product/sr/Pages/toto_results.aspx
  https://www.singaporepools.com.sg/DataFileArchive/Lottery/Output/toto_result_top_draws_en.html
