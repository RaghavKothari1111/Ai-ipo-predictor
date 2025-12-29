
import difflib

name1 = "Gujarat Kidney"
name2 = "Gujarat Kidney and Super Speciality"

ratio = difflib.SequenceMatcher(None, name1, name2).ratio()
print(f"Ratio: {ratio}")
matches = difflib.get_close_matches(name1, [name2], n=1, cutoff=0.6)
print(f"Match with 0.6: {matches}")
matches_05 = difflib.get_close_matches(name1, [name2], n=1, cutoff=0.5)
print(f"Match with 0.5: {matches_05}")
