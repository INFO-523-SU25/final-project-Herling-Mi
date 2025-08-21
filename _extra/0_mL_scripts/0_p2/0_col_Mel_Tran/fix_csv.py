import pandas as pd
import os

# Paths
input_csv = r"C:\Users\Mr_Green\OneDrive\Desktop\0_Sum_25\INFO_523\0_Final_Proj_N\0_grey_Mel_Train\output_sorted_with_grey.csv"
output_csv = os.path.join(os.getcwd(), "output_sorted_with_viridis.csv")

# Read CSV with UTF-8-sig
df = pd.read_csv(input_csv, encoding='utf-8-sig')

# Rename column
df.rename(columns={"grey_png": "color_png"}, inplace=True)

# Correctly update paths
df["color_png"] = df["color_png"].str.replace(
    r"_wav_grey_mel", "_wav_viridis_mel", regex=False
).str.replace(
    "_gray.png", "_viridis.png", regex=False
)

# Save updated CSV with UTF-8-sig
df.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f"Updated CSV saved as: {output_csv}")
