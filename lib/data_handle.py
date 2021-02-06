"""
Load data from local path
"""
import csv

data = """
"""


def load_data(path: str):
    result = []
    with open(path, 'r') as f:
        raw = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for d in raw:
            result += d
    return result


def save_data():
    pass
