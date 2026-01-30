import os
import json

# project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_wage_bounds():
    """Load wage bounds from JSON file."""
    # load data
    path = os.path.join(_PROJECT_ROOT, "data", "wage_bounds.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # iterate through oecd-based and dataset-based
    for version in ("oecd_based", "dataset_based"):
        block = data.get(version)
        if isinstance(block, dict):  # verify each block is a dictionary
            try:  # try unpacking oecd-based first
                min_wage = int(block["min_monthly_wage"])  # unpack min wage
                max_wage = int(block["max_monthly_wage"])  # unpack max wage
                return min_wage, max_wage  # return min and max wages
            except:  # if error, unpack dataset-based
                continue  # continue to next block
    raise ValueError("wage_bounds.json: no valid oecd_based or dataset_based")