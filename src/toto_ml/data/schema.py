"""Column schema. Only these columns are used; anything else in the file is ignored."""

MAIN_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]
ADDITIONAL_COL = "additional"
REQUIRED_COLS = ["draw_date"] + MAIN_COLS + [ADDITIONAL_COL]
# draw_number is optional; when absent it is generated from chronological order (1..N).
OUTPUT_COLS = ["draw_number"] + REQUIRED_COLS
