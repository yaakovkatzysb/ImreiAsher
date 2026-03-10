"""Search for specific words in the JSON output to debug missing words."""
import json
import sys

SEARCH_WORDS = {"שרואים", "להיות", "באחדות", "שצריך"}

path = "../data/output/json/548.json"
data = json.load(open(path, "r", encoding="utf-8"))
pages = data.get("pages", data.get("pages_data", []))

for p in pages:
    page_num = p.get("page_number", "?")
    words = p.get("words_data", [])
    for w in words:
        if w["text"] in SEARCH_WORDS:
            bbox = w.get("bbox", [])
            ys = [pt[1] for pt in bbox] if bbox else []
            y_center = sum(ys) / len(ys) if ys else 0
            print(f"  page={page_num}  word='{w['text']}'  y_center={y_center:.0f}  bbox={bbox}")
