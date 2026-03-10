"""Search for missing words in final text and Google Vision cache."""
import json
import os
from pathlib import Path

SEARCH_WORDS = ["שרואים", "להיות", "באחדות", "שצריך"]

# 1. Check final text output
print("=== 1. חיפוש בטקסט הסופי ===")
txt_path = "../data/output/txt/548.txt"
if os.path.exists(txt_path):
    text = open(txt_path, "r", encoding="utf-8").read()
    for word in SEARCH_WORDS:
        if word in text:
            # Find the line containing this word
            for i, line in enumerate(text.splitlines(), 1):
                if word in line:
                    print(f"  '{word}' נמצא בשורה {i}: {line.strip()[:80]}")
        else:
            print(f"  '{word}' *** לא נמצא בטקסט הסופי! ***")
else:
    print(f"  קובץ לא נמצא: {txt_path}")

# 2. Check Google Vision cache
print("\n=== 2. חיפוש בקאש Google Vision ===")
cache_dir = Path("../data/cache")
if cache_dir.exists():
    cache_files = sorted(cache_dir.glob("*548*"))
    print(f"  קבצי קאש: {len(cache_files)}")
    for cf in cache_files:
        print(f"    {cf.name} ({cf.stat().st_size} bytes)")
        try:
            data = json.load(open(cf, "r", encoding="utf-8"))
            # Search in cached words
            words = data.get("words", [])
            for w in words:
                if isinstance(w, dict) and w.get("text") in SEARCH_WORDS:
                    bbox = w.get("bbox", [])
                    ys = [pt[1] for pt in bbox] if bbox else []
                    y_center = sum(ys) / len(ys) if ys else 0
                    print(f"      word='{w['text']}'  y={y_center:.0f}  bbox={bbox}")
        except Exception as e:
            print(f"      error: {e}")
else:
    print(f"  תיקיית קאש לא נמצאה: {cache_dir}")
