import logging
import os


def setup_logging(level: str | None = None) -> None:
    level = level or os.environ.get("TOTO_ML_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    for noisy in ("matplotlib", "numba", "urllib3", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
