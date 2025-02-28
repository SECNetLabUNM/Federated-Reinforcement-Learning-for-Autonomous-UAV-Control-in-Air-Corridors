import bibtexparser
from collections import defaultdict

# Load your BibTeX file
with open('your_bibtex_file.bib') as bibtex_file:
    bib_database = bibtexparser.load(bibtex_file)

# Dictionary to track entries by title
entries_by_title = defaultdict(list)

# Populate the dictionary
for entry in bib_database.entries:
    title = entry.get('ID', '').strip().lower()
    if title not in entries_by_title.keys():
        entries_by_title[title].append(entry)
# Filter out duplicates by keeping the first occurrence of each title
unique_entries = [entries[0] for entries in entries_by_title.values()]  # Flattening the list of lists to a single list of entries

# Assign the list of unique entries back to the BibTeX database
bib_database.entries = unique_entries

# Save the new BibTeX file
with open('deduplicated_bibtex_file.bib', 'w') as bibtex_file:
    bibtexparser.dump(bib_database, bibtex_file)

print("Duplicates have been removed.")
