"""Regenerate the static JSON payload used by GitHub Pages and Vercel."""

import json
import os

from server import DATA


OUTPUT = os.path.join(os.path.dirname(__file__), "..", "data.json")
with open(OUTPUT, "w", encoding="utf-8") as output_file:
    json.dump(DATA, output_file, ensure_ascii=False, separators=(",", ":"))
    output_file.write("\n")

print(f"Wrote {DATA['total_tx']:,} transactions and {len(DATA['rfm_customers']):,} customer profiles to {OUTPUT}")
