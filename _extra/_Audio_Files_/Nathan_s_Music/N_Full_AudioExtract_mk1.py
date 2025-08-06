import yt_dlp
from pydub import AudioSegment
import os
import re
import time
import multiprocessing
import sys
import pandas as pd
import ast
import json

# === CONFIGURATION ===
CSV_INPUT_PATH = r"Nathans_Music_list_processed_complete_with_YT_links.csv"
MP3_FOLDER = "mp3"
WAV_FOLDER = "wav"
TIMEOUT_SECONDS = 120
PROCESS_LIMIT = None  # Set to None to process all rows
OUTPUT_CSV_PATH = "N_Full_db_Mk1.csv"
OUTPUT_JSON_PATH = "N_Full_db_Mk1.json"

# === HELPERS ===

def sanitize_filename(name):
    """Allow valid Unicode characters, but remove or replace invalid filesystem characters."""
    return re.sub(r'[\\/*?:"<>|]', '_', name)

def download_audio_yt_dlp(url, output_folder):
    """Download best audio from YouTube and convert to MP3."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_folder, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }

    # Suppress all yt-dlp/ffmpeg output
    sys_stdout, sys_stderr = sys.stdout, sys.stderr
    sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    sys.stderr = open(os.devnull, 'w', encoding='utf-8')

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    finally:
        sys.stdout = sys_stdout
        sys.stderr = sys_stderr

    video_id = info.get("id")
    title = info.get("title", video_id)
    sanitized_title = sanitize_filename(title)

    temp_mp3 = os.path.join(output_folder, f"{video_id}.mp3")
    final_mp3 = os.path.join(output_folder, f"{sanitized_title}.mp3")

    # Resolve duplicate filename collisions
    if temp_mp3 != final_mp3:
        if os.path.exists(final_mp3):
            base, ext = os.path.splitext(final_mp3)
            count = 1
            while os.path.exists(f"{base}_{count}{ext}"):
                count += 1
            final_mp3 = f"{base}_{count}{ext}"
        os.rename(temp_mp3, final_mp3)

    return final_mp3, sanitized_title

def convert_mp3_to_wav(mp3_path, output_folder, base_filename):
    """Convert MP3 to WAV format and return the WAV path."""
    wav_path = os.path.join(output_folder, f"{base_filename}.wav")
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format='wav')
    return wav_path

def process_single_url(url, mp3_folder, wav_folder, return_dict):
    """Download and convert one YouTube link. Store results in return_dict."""
    try:
        start = time.time()
        mp3_path, base_filename = download_audio_yt_dlp(url, mp3_folder)
        wav_path = convert_mp3_to_wav(mp3_path, wav_folder, base_filename)
        elapsed = time.time() - start

        return_dict['success'] = True
        return_dict['mp3'] = os.path.abspath(mp3_path)
        return_dict['wav'] = os.path.abspath(wav_path)
        return_dict['title'] = base_filename
        return_dict['elapsed'] = elapsed
    except Exception as e:
        return_dict['error'] = str(e)

def ensure_folders():
    """Create output folders if they do not exist."""
    os.makedirs(MP3_FOLDER, exist_ok=True)
    os.makedirs(WAV_FOLDER, exist_ok=True)

def ensure_columns(df, columns):
    """Ensure DataFrame columns exist and are of string-compatible dtype."""
    for col in columns:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)

# === MAIN FUNCTION ===

def main():
    # Load CSV input
    df = pd.read_csv(CSV_INPUT_PATH, encoding='utf-8')

    # Ensure folders and column structure
    ensure_folders()
    ensure_columns(df, ['File_mp3', 'File_wav'])

    # Parse links from the YT_link column
    parsed_rows = []
    for idx, row in df.iterrows():
        try:
            links = ast.literal_eval(row['YT_link'])
            if isinstance(links, list):
                parsed_rows.append((idx, links))
        except Exception as e:
            print(f"⚠️  Skipping row {idx}: {e}")

    # Limit processing (optional)
    if PROCESS_LIMIT is not None:
        parsed_rows = parsed_rows[:PROCESS_LIMIT]

    total_start = time.time()
    metadata_list = []

    for row_index, links in parsed_rows:
        mp3_paths = []
        wav_paths = []

        for url in links:
            print(f"\n🔄 Processing: {url}")
            manager = multiprocessing.Manager()
            return_dict = manager.dict()

            proc = multiprocessing.Process(
                target=process_single_url,
                args=(url, MP3_FOLDER, WAV_FOLDER, return_dict)
            )
            proc.start()
            proc.join(timeout=TIMEOUT_SECONDS)

            metadata = {
                "yt_url": url,
                "title": return_dict.get("title"),
                "mp3": return_dict.get("mp3"),
                "wav": return_dict.get("wav"),
                "elapsed_seconds": return_dict.get("elapsed"),
                "error": return_dict.get("error") if not return_dict.get("success") else None,
            }

            if proc.is_alive():
                print("⏰ Timeout: Skipping...")
                proc.terminate()
                proc.join()
                metadata["error"] = "Timeout"

            elif return_dict.get('success'):
                print(f"✅ Title: {return_dict['title']}")
                print(f"   MP3: {return_dict['mp3']}")
                print(f"   WAV: {return_dict['wav']}")
                print(f"   ⏱️ Time: {return_dict['elapsed']:.2f} sec")
                mp3_paths.append(return_dict['mp3'])
                wav_paths.append(return_dict['wav'])
            else:
                print(f"❌ Error: {return_dict.get('error', 'Unknown error')}")

            metadata_list.append(metadata)

        # Store resulting file paths in the DataFrame
        df.at[row_index, 'File_mp3'] = str(mp3_paths)
        df.at[row_index, 'File_wav'] = str(wav_paths)

    total_end = time.time()
    print(f"\n🕒 Total time: {total_end - total_start:.2f} seconds")

    # Save updated CSV with UTF-8 encoding
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"\n📄 CSV saved to: {OUTPUT_CSV_PATH}")

    # Save metadata to JSON
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as jf:
        json.dump(metadata_list, jf, indent=2, ensure_ascii=False)

    print(f"📝 JSON metadata saved to: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
