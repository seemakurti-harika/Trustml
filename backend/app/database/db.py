import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "trustml.db"

def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with connect() as c:
        c.execute('''CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY, created_at TEXT, dataset_name TEXT, rows_count INTEGER,
            feature_count INTEGER, target TEXT, model_name TEXT, parameters TEXT, metrics TEXT, status TEXT)''')
