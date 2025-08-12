import pandas as pd
import json
import os
import re

# === File & Folder Setup ===
input_file = "/Users/mq/Desktop/INFO523/Herling-Mi/_extra/_Audio_Files_/Yashi_s_Music/Yashis_Music_list.xlsx"
output_folder = "Yashis_processed_music_list"
os.makedirs(output_folder, exist_ok=True)

# Output file paths
output_excel = os.path.join(output_folder, "Yashis_Music_list_processed.xlsx")
output_csv = os.path.join(output_folder, "Yashis_Music_list_processed.csv")
output_json = os.path.join(output_folder, "Yashis_Music_list_processed.json")

# === Region Parsing Helper ===
def parse_regions(region_series):
    split_regions = region_series.dropna().apply(lambda r: re.split(r'[,/]', str(r)))
    flat = [r.strip() for sublist in split_regions for r in sublist if r.strip()]
    return sorted(set(flat))  # Remove duplicates, sort alphabetically

# === Read Input Excel File ===
df = pd.read_excel(input_file)

# === Group and Aggregate Data by Artist ===
grouped_df = df.groupby("Artist").apply(
    lambda g: pd.Series({
        "Songs": g["Song"].tolist(),
        "Categories": g["Category"].tolist(),
        "Region": parse_regions(g["Region"]),
    })
).reset_index()

# === Add 'User' Column at the Beginning ===
grouped_df.insert(0, "User", "Yahis")

# === Add 'YT_link' Column ===
grouped_df["YT_link"] = grouped_df["Songs"].apply(lambda songs: [None] * len(songs))

# === Export to Excel ===
grouped_df.to_excel(output_excel, index=False)

# === Export to CSV (comma-separated Region string for readability) ===
grouped_df["Region"] = grouped_df["Region"].apply(lambda lst: ", ".join(lst))
grouped_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

# === Export to JSON ===
grouped_df["Region"] = grouped_df["Region"].apply(lambda s: s.split(", "))
grouped_df.to_json(output_json, orient="records", force_ascii=False, indent=4)

# === Done ===
print("✅ Refactor complete. Files saved to:")
print(f"📄 {output_excel}")
print(f"📄 {output_csv}")
print(f"📄 {output_json}")
