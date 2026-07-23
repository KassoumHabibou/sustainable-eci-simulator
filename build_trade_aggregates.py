"""
build_trade_aggregates.py — Compact trade panels for the simulator app
======================================================================
Aggregates the bilateral OEC files (exporter, importer, hs_code, value)
into two small parquet files consumed by the Streamlit app:

  data/trade_exports.parquet : year × country × hs4 → export value (USD)
  data/trade_imports.parquet : year × country × hs4 → import value (USD)

Run once on a machine that has the raw bilateral files
(../new_approach/input/data/trade_data/bilateral_YYYY.csv):
    python build_trade_aggregates.py
"""

import glob
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(os.path.dirname(ROOT), "new_approach",
                       "input", "data", "trade_data")
OUT_EXP = os.path.join(ROOT, "data", "trade_exports.parquet")
OUT_IMP = os.path.join(ROOT, "data", "trade_imports.parquet")

exp_parts, imp_parts = [], []
files = sorted(glob.glob(os.path.join(RAW_DIR, "bilateral_*.csv")))
if not files:
    raise SystemExit(f"No bilateral files found in {RAW_DIR}")

for f in files:
    year = int(re.search(r"(\d{4})", os.path.basename(f)).group(1))
    df = pd.read_csv(f, usecols=["exporter_id", "importer_id",
                                 "hs_code", "value"],
                     dtype={"hs_code": str})
    e = (df.groupby(["exporter_id", "hs_code"], as_index=False)["value"]
         .sum().rename(columns={"exporter_id": "country"}))
    i = (df.groupby(["importer_id", "hs_code"], as_index=False)["value"]
         .sum().rename(columns={"importer_id": "country"}))
    for part in (e, i):
        part["year"] = year
        part["value"] = part["value"].astype("float32")
    exp_parts.append(e)
    imp_parts.append(i)
    print(f"{year}: exports {len(e):,} rows, imports {len(i):,} rows")

for parts, out in ((exp_parts, OUT_EXP), (imp_parts, OUT_IMP)):
    full = pd.concat(parts, ignore_index=True)
    full["country"] = full["country"].astype("category")
    full["hs_code"] = full["hs_code"].astype("category")
    full["year"] = full["year"].astype("int16")
    full.to_parquet(out, index=False, compression="zstd")
    print(f"{out}: {len(full):,} rows, {os.path.getsize(out)/1e6:.1f} MB")
