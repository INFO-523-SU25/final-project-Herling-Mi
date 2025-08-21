import pandas as pd
from pathlib import Path

# -----------------------------
# 1. Read CSV
# -----------------------------
csv_file = r"C:\Users\Mr_Green\OneDrive\Desktop\0_Sum_25\INFO_523\0_Final_Proj_N\0_grey_Mel_Train\N_audio_features_parallel_output_encoded.csv"

try:
    df = pd.read_csv(csv_file, encoding="utf-8-sig")
except UnicodeDecodeError:
    # fallback if utf-8-sig fails
    df = pd.read_csv(csv_file, encoding="latin1")

# -----------------------------
# 2. Sort CSV by artist & song_name
# -----------------------------
df_sorted = df.sort_values(by=["artist", "song_name"], ascending=[True, True]).reset_index(drop=True)

# -----------------------------
# 3. List PNG files from folders
# -----------------------------
# Grey PNGs
grey_folder = Path(r"C:\Users\Mr_Green\OneDrive\Desktop\0_Sum_25\INFO_523\0_Final_Proj_N\wav_files\_wav_grey_mel")
grey_files = sorted([str(p.resolve()) for p in grey_folder.glob("*.png")])

# Viridis PNGs
viridis_folder = Path(r"C:\Users\Mr_Green\OneDrive\Desktop\0_Sum_25\INFO_523\0_Final_Proj_N\wav_files\_wav_viridis_mel")
viridis_files = sorted([str(p.resolve()) for p in viridis_folder.glob("*.png")])

# -----------------------------
# 4. Add columns to DataFrame
# -----------------------------
if len(grey_files) != len(df_sorted):
    print(f"Warning: Number of Grey PNG files ({len(grey_files)}) != number of CSV rows ({len(df_sorted)})")
if len(viridis_files) != len(df_sorted):
    print(f"Warning: Number of Viridis PNG files ({len(viridis_files)}) != number of CSV rows ({len(df_sorted)})")

df_sorted["grey_png"] = grey_files[:len(df_sorted)]       # truncate if mismatch
df_sorted["viridis_png"] = viridis_files[:len(df_sorted)] # truncate if mismatch

# -----------------------------
# 5. Write out CSV with utf-8-sig
# -----------------------------
output_file = r"C:\Users\Mr_Green\OneDrive\Desktop\0_Sum_25\INFO_523\0_Final_Proj_N\0_grey_Mel_Train\output_sorted_with_grey_viridis.csv"
df_sorted.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"Done! CSV saved to: {output_file}")
