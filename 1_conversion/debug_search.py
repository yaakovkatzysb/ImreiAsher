"""Search for specific words in the JSON output to debug missing words."""
import json

SEARCH_WORDS = {"שרואים", "להיות", "באחדות", "שצריך"}

path = "../data/output/json/548.json"
data = json.load(open(path, "r", encoding="utf-8"))

# Show top-level keys to understand structure
print("Top-level keys:", list(data.keys()))
for k, v in data.items():
    print(f"  {k}: type={type(v).__name__}, ", end="")
    if isinstance(v, list):
        print(f"len={len(v)}")
        if v and isinstance(v[0], dict):
            print(f"    first item keys: {list(v[0].keys())}")
    elif isinstance(v, dict):
        print(f"keys={list(v.keys())}")
    else:
        print(f"value={str(v)[:80]}")

# Try to find pages with words_data
def find_words(obj, depth=0):
    if isinstance(obj, dict):
        if "words_data" in obj and isinstance(obj["words_data"], list):
            page = obj.get("page_number", "?")
            for w in obj["words_data"]:
                if isinstance(w, dict) and w.get("text") in SEARCH_WORDS:
                    bbox = w.get("bbox", [])
                    ys = [pt[1] for pt in bbox] if bbox else []
                    y_center = sum(ys) / len(ys) if ys else 0
                    print(f"  page={page}  word='{w['text']}'  y_center={y_center:.0f}  bbox={bbox}")
        for v in obj.values():
            find_words(v, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            find_words(item, depth + 1)

print("\n--- Searching for words ---")
find_words(data)
