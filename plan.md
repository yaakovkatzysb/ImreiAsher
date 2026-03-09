# Plan: Flag Ambiguous OCR Words with ⚠️

## Problem
Google Vision OCR sometimes confuses visually similar Hebrew letters (e.g., נ↔כ, ד↔ר).
When both the original and the swapped version are valid words/abbreviations, there's no way to auto-correct.
Example: כמש"כ vs כמש"נ — both are valid abbreviations.

## Solution
Add inline ⚠️ markers around ambiguous words in the text output so the user can review them manually.

## Changes

### 1. `dictionary_checker.py` — New method `flag_ambiguous_words()`
- Iterate over all tokens in the text
- For each known word, try all single-letter confusion swaps
- If any swap also produces a known word → wrap with ⚠️: `⚠️כמש"כ⚠️`
- If symbol_confidences available and confidence > 0.95 for the confused letter → skip flagging (OCR is confident)
- Return the modified text

### 2. `main.py` — Call `flag_ambiguous_words()` in pipeline
- After Step 3c (manual corrections), call `flag_ambiguous_words()` on each page's text
- Pass `words_data` for confidence-based filtering
- Also wire up `fix_confused_letters()` which is currently unused

## What stays the same
- Text output format (.txt) — just adds ⚠️ markers where needed
- JSON output — contains the flagged text
- No changes to OCR, preprocessing, or post-processing logic
