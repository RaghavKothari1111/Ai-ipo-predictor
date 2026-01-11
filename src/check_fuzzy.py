
import difflib

name1 = "Bharat Coking Coal"
name2 = "Bharat Coking Coal"  # After parse_name removes 'Ltd.'

ratio = difflib.SequenceMatcher(None, name1, name2).ratio()
print(f"Exact match ratio: {ratio}")

name2_with_ltd = "Bharat Coking Coal Ltd."
ratio2 = difflib.SequenceMatcher(None, name1, name2_with_ltd).ratio()
print(f"Ratio with Ltd.: {ratio2}")

matches = difflib.get_close_matches(name1, [name2_with_ltd], n=1, cutoff=0.55)
print(f"Match with 0.55: {matches}")
