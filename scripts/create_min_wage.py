"""
Build wage_bounds.json with two versions:
- dataset_based: min and max from project CSV (Monthly_Salary).
- oecd_based: min from OECD API, max from project CSV.

Run:
- python -m scripts.create_min_wage
- python ./scripts/create_min_wage.py

Output:
- data/wage_bounds.json
"""

import json
import os
import sys
from io import StringIO

import requests
import pandas as pd

# project root directory (./data/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# project CSV
CSV_NAME = "Extended_Employee_Performance_and_Productivity_Data.csv"

# salary column
SALARY_COL = "Monthly_Salary"

# OECD API (CSV with labels, from 2010)
OECD_MINWAGE_URL = (
    "https://sdmx.oecd.org/public/rest/data/OECD.ELS.SAE,DSD_EARNINGS@MW_CURP,1.0/"
    "..USD.A...?startPeriod=2010&dimensionAtObservation=AllDimensions&format=csvfilewithlabels"
)

# columns to drop
COLS_TO_DROP = [
    "STRUCTURE", "STRUCTURE_ID", "STRUCTURE_NAME", "ACTION",
    "Reference area", "MEASURE", "Unit of measure", "PAY_PERIOD", "PRICE_BASE",
    "AGGREGATION_OPERATION", "Aggregation operation", "SEX", "Sex",
    "Time period", "Observation value", "BASE_PER", "Base period",
    "OBS_STATUS", "Observation status", "UNIT_MULT", "Unit multiplier",
    "DECIMALS", "Decimals",
]

########################################################################################################

def _dataset_bounds():
    """Min and max monthly salary from project CSV."""
    path = os.path.join(PROJECT_ROOT, "data", CSV_NAME)
    df = pd.read_csv(path, usecols=[SALARY_COL])
    s = pd.to_numeric(df[SALARY_COL], errors="coerce").dropna()
    min_wage = int(s.min())
    global max_wage  # global variable
    max_wage = int(s.max())
    return (min_wage, max_wage)

########################################################################################################

def _oecd_bounds():
    """Min monthly salary from OECD average yearly minimum wage."""
    response = requests.get(OECD_MINWAGE_URL, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns], axis=1, inplace=True)
    avg_yearly = float(pd.to_numeric(df["OBS_VALUE"], errors="coerce").dropna().mean())
    min_wage = int(round(avg_yearly / 12, 0))
    global max_wage  # reuse global variable
    return (min_wage, max_wage)

########################################################################################################

def output_JSON():
    """Write JSON file with dataset-based and oecd-based."""
    out_path = os.path.join(PROJECT_ROOT, "data", "wage_bounds.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # get dataset-based bounds
    d_min, d_max = _dataset_bounds()
    # create payload
    payload = {
        "dataset_based": {
            "min_monthly_wage": d_min,
            "max_monthly_wage": d_max,
        },
    }

    # try oecd-based bounds
    try:
        oecd_min, oecd_max = _oecd_bounds()
        # create payload
        payload["oecd_based"] = {
            "min_monthly_wage": oecd_min,
            "max_monthly_wage": oecd_max,
        }
    except Exception:
        pass  # OECD failed; only dataset_based is written

    # write to file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return True

########################################################################################################

if __name__ == "__main__":
    try:
        ok = output_JSON()
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    sys.exit(0 if ok else 1)