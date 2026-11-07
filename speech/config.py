import os
from datetime import date

# Set CREC_DATA_DIR in the environment, or edit the default below.
DATA_DIR = os.environ.get("CREC_DATA_DIR", "/n/path/to/your/crec_data")
RAW_DIR = os.path.join(DATA_DIR, "raw", "output")   # downloader writes here
OUT_DIR = os.path.join(DATA_DIR, "parsed")          # parsed HDF5 files go here

# Congresses start January 3 of odd years. Ends are inclusive of that date,
# and parsing keeps only rows whose metadata says the right Congress number.
CONGRESS_DATES = {
    111: ("2009-01-03", "2011-01-03"),
    112: ("2011-01-03", "2013-01-03"),
    113: ("2013-01-03", "2015-01-03"),
    114: ("2015-01-03", "2017-01-03"),
    115: ("2017-01-03", "2019-01-03"),
    116: ("2019-01-03", "2021-01-03"),
    117: ("2021-01-03", "2023-01-03"),
    118: ("2023-01-03", "2025-01-03"),
    119: ("2025-01-03", date.today().isoformat()),  # ongoing
}

def date_range(congress):
    start, end = CONGRESS_DATES[congress]
    return start, min(end, date.today().isoformat())
