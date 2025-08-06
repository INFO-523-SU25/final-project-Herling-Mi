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
INPUT_CSV = "Yashis_Music_list_processed.csv"
OUTPUT_CSV = "Yashis_Music_list_processed_complete_with_YT_links.csv"

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
    df = pd.read_csv(INPUT_CSV)

    # If YT_link is not present or empty, initialize with [None, None, ...]
    if 'YT_link' not in df.columns or df['YT_link'].isnull().all():
        df['YT_link'] = df['Songs'].apply(lambda s: str([None] * len(ast.literal_eval(s))))

    driver = setup_driver()

    try:
        for idx, row in df.iterrows():
            if MAX_ROWS is not None and idx >= MAX_ROWS:
                print(f"🛑 Stopping after {MAX_ROWS} rows (test mode).")
                break

            artist = row['Artist']
            try:
                songs = ast.literal_eval(row['Songs'])
            except:
                print(f"⚠️ Could not parse Songs for row {idx}: {row['Songs']}")
                continue

            yt_links = []
            print(f"\n🎵 {artist}")
            for song in songs:
                print(f"🔍 Searching: {artist} - {song}")
                try:
                    link = search_youtube_second_link_only(driver, artist, song)
                    yt_links.append(link)
                    print(f"   → ✅ {link}")
                except Exception as e:
                    print(f"   → ❌ Error: {e}")
                    yt_links.append("")

                time.sleep(1)  # Polite delay

            df.at[idx, 'YT_link'] = json.dumps(yt_links)  # Store as JSON string for CSV compatibility

    finally:
        driver.quit()

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ Done. Output written to: {OUTPUT_CSV}")

# === RUN ===
if __name__ == "__main__":
    update_csv_with_youtube_links()
