import os
import re
import csv
from dateutil import parser

# Folder containing news articles organized in source-specific subfolders
base_path = "data/MC1/News Articles"
output_csv = "news_articles_metadata.csv"

# === METADATA HELPERS ===

# Extract metadata field (e.g. TITLE, AUTHOR, etc.) using regex
def extract_metadata(field, text):
    pattern = rf"{field}:\s*(.*)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else ""

# Escape quotes and remove line breaks for clean CSV output
def safe(text):
    if not isinstance(text, str):
        return ""
    return text.replace('"', '""').replace("\n", " ").replace("\r", " ").strip()

# === PROCESS ARTICLES AND WRITE CSV ===

with open(output_csv, mode="w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(["filename", "source", "title", "date", "author", "location", "content"])

    for root, _, files in os.walk(base_path):
        for file in files:
            if not file.endswith(".txt"):
                continue

            filepath = os.path.join(root, file)
            source = os.path.basename(root)

            with open(filepath, encoding="utf-8", errors="ignore") as txt_file:
                text = txt_file.read()

                # Special case: The Abila Post has inconsistent formatting
                if source.lower() == "the abila post":
                    match = re.match(r"^\s*(\d{4}[-/]\d{2}[-/]\d{2})\s*[–-]", text)
                    if match:
                        raw_date = match.group(1)
                        try:
                            parsed_date = parser.parse(raw_date)
                            date = parsed_date.strftime("%Y/%m/%d")
                        except Exception:
                            date = ""
                        # Remove the date line from the top of the article
                        text = text[match.end():].lstrip()
                    else:
                        date = ""
                else:
                    raw_date = extract_metadata("PUBLISHED", text)
                    try:
                        parsed_date = parser.parse(raw_date, fuzzy=True)
                        date = parsed_date.strftime("%Y/%m/%d")
                    except Exception:
                        date = ""

            # Extract metadata fields
            title = extract_metadata("TITLE", text)
            author = extract_metadata("AUTHOR", text)

            # Fallback: extract author from "PUBLISHED" line if missing
            if source.lower() == "the abila post" and not author:
                pub_line = extract_metadata("PUBLISHED", text)
                match = re.match(r"By ([\w\s.]+)", pub_line)
                if match:
                    author = match.group(1).strip()

            # Try to extract location from the top 500 characters only
            metadata_section = text[:500]
            location = extract_metadata("LOCATION", metadata_section)

            # Re-attempt to extract date if not already found
            raw_date = extract_metadata("PUBLISHED", text)
            if raw_date:
                try:
                    parsed_date = parser.parse(raw_date, fuzzy=True)
                    date = parsed_date.strftime("%Y/%m/%d")
                except Exception:
                    date = ""

            # Fallback: find a loose date at the start of the article
            if not date and source.lower() == "the abila post":
                date_match = re.search(r"\b(\d{1,2}\s+\w+\s+\d{4}|\d{4}[-/]\d{2}[-/]\d{2})\b", text)
                if date_match:
                    try:
                        parsed_date = parser.parse(date_match.group(1), fuzzy=True)
                        date = parsed_date.strftime("%Y/%m/%d")
                        # Remove the matched date from the article
                        text = text.replace(date_match.group(0), "", 1).strip()
                    except Exception:
                        date = ""

            # Remove metadata headers from article body
            content_cleaned = re.sub(r"(?mi)^(SOURCE|TITLE|PUBLISHED|AUTHOR|LOCATION):.*$", "", text)
            content = content_cleaned.replace("\n", " ").replace("\r", " ").strip()

            # Write processed row to CSV
            writer.writerow([
                safe(file),
                safe(source),
                safe(title),
                safe(date),
                safe(author),
                safe(location),
                safe(content)
            ])

print(f"Done, saved as: {output_csv}")