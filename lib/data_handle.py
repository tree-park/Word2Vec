"""
Load data from local path
"""
import csv

data = """
"""


def load_data(path: str):
    result = []
    with open(path, 'r') as f:
        raw = csv.reader(f)
        for d in raw:
            result += d
    return result


def save_data():
    pass
