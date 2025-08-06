import csv
import json
import time
import ast
import urllib.parse
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# === CONTROL VARIABLE ===
MAX_ROWS = None  # Change to None for full run

# === INPUT / OUTPUT ===
INPUT_CSV = "Nathans_Music_list_processed.csv"
OUTPUT_CSV = "Nathans_Music_list_processed_complete_with_YT_links.csv"

# === EXPECTED CSV KEYS ===
REQUIRED_COLUMNS = ['User', 'Artist', 'Songs', 'Category', 'Region', 'YT_link']

# === SELENIUM SETUP ===
def setup_driver():
    options = Options()
    options.headless = True
    return webdriver.Chrome(options=options)

def search_youtube_second_link_only(driver, artist, song):
    query = f"{artist} {song}"
    search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
    driver.get(search_url)
    time.sleep(2)

    video_links = driver.find_elements(By.XPATH, '//a[starts-with(@href, "/watch?v=")]')
    seen = set()
    cleaned_links = []

    for link in video_links:
        href = link.get_attribute('href')
        if href:
            parsed = urllib.parse.urlparse(href)
            query_params = urllib.parse.parse_qs(parsed.query)
            video_id = query_params.get("v", [None])[0]
            if video_id:
                clean_url = f"https://www.youtube.com/watch?v={video_id}"
                if clean_url not in seen:
                    seen.add(clean_url)
                    cleaned_links.append(clean_url)
        if len(cleaned_links) >= 2:
            break

    return cleaned_links[1] if len(cleaned_links) >= 2 else ""

# === MAIN PROCESSING ===
def update_csv_with_youtube_links():
    print(f"Loading CSV file: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    print("Original columns found:", df.columns.tolist())

    # Strip whitespace from columns
    df.columns = df.columns.str.strip()
    print("Columns after stripping whitespace:", df.columns.tolist())

    # Check for missing columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"❌ ERROR: Missing required columns: {missing_cols}")
        print("Please check your CSV header row for typos or extra spaces.")
        return

    # Initialize 'YT_link' column if missing or empty
    if 'YT_link' not in df.columns or df['YT_link'].isnull().all():
        def init_yt_links(songs_str):
            try:
                songs = ast.literal_eval(songs_str)
                return json.dumps([None] * len(songs))
            except Exception as e:
                print(f"⚠️ Could not parse Songs to initialize YT_link: {songs_str} - {e}")
                return json.dumps([])

        df['YT_link'] = df['Songs'].apply(init_yt_links)

    driver = setup_driver()

    try:
        for idx, row in df.iterrows():
            if MAX_ROWS is not None and idx >= MAX_ROWS:
                print(f"🛑 Stopping after {MAX_ROWS} rows (test mode).")
                break

            artist = row['Artist']

            # Parse songs list
            try:
                songs = ast.literal_eval(row['Songs'])
            except Exception as e:
                print(f"⚠️ Could not parse Songs for row {idx}: {row['Songs']} - {e}")
                continue

            # Parse existing YT_link list or initialize
            try:
                yt_links = ast.literal_eval(row['YT_link'])
                # Reset if length mismatch or empty
                if len(yt_links) != len(songs):
                    yt_links = [None] * len(songs)
            except Exception as e:
                print(f"⚠️ Could not parse YT_link for row {idx}, resetting: {e}")
                yt_links = [None] * len(songs)

            print(f"\n🎵 Artist: {artist}")

            # Search YouTube for songs without a link
            for i, song in enumerate(songs):
                if yt_links[i]:
                    print(f"🔍 Skipping (already has link): {artist} - {song}")
                    continue

                print(f"🔍 Searching: {artist} - {song}")
                try:
                    link = search_youtube_second_link_only(driver, artist, song)
                    yt_links[i] = link
                    print(f"   → ✅ Found link: {link}")
                except Exception as e:
                    print(f"   → ❌ Error searching for {artist} - {song}: {e}")
                    yt_links[i] = ""

                time.sleep(1)  # Polite delay

            df.at[idx, 'YT_link'] = json.dumps(yt_links)

    finally:
        driver.quit()

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ Done. Output written to: {OUTPUT_CSV}")

# === RUN ===
if __name__ == "__main__":
    update_csv_with_youtube_links()
