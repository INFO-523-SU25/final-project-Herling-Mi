import os
import json
import re
import pandas as pd
from yt_dlp import YoutubeDL
from pydub import AudioSegment

# === INPUT/OUTPUT CONFIG ===
INPUT_CSV = r"C:\Users\Mr_Green\OneDrive\Desktop\0_Sum_25\INFO_523\HWs\group_project\Yashis_Suite\Final_org_attempt_mk1\Nathans_processed_music_list\Nathans_Music_list_processed_complete_with_YT_links.csv"
OUTPUT_CSV = "Nathans_complete_data.csv"
AUDIO_DIR = "downloaded_audio"
MP3_DIR = os.path.join(AUDIO_DIR, "mp3")
WAV_DIR = os.path.join(AUDIO_DIR, "wav")

# === CREATE DIRECTORIES ===
os.makedirs(MP3_DIR, exist_ok=True)
os.makedirs(WAV_DIR, exist_ok=True)

# === FILENAME SANITIZATION ===
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', '_', name)

# === DOWNLOAD AND CONVERT FUNCTIONS ===
def download_audio_yt_dlp(url, output_folder):
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

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id")
        title = info.get("title", video_id)
        mp3_temp_path = os.path.join(output_folder, f"{video_id}.mp3")

        # Sanitize title and make final filename
        safe_title = sanitize_filename(title)
        mp3_final_path = os.path.join(output_folder, f"{safe_title}.mp3")

        # Prevent overwrite
        if mp3_temp_path != mp3_final_path:
            if os.path.exists(mp3_final_path):
                base, ext = os.path.splitext(mp3_final_path)
                count = 1
                while os.path.exists(f"{base}_{count}{ext}"):
                    count += 1
                mp3_final_path = f"{base}_{count}{ext}"
            os.rename(mp3_temp_path, mp3_final_path)

    return mp3_final_path, safe_title

def convert_mp3_to_wav(mp3_path, output_folder, base_filename):
    wav_path = os.path.join(output_folder, f"{base_filename}.wav")
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format='wav')
    return wav_path

# === PROCESSING CSV ===
def process_csv():
    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip()

    # Ensure required columns exist
    required_cols = ['YT_link', 'File_mp3', 'File_wav']
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    for idx, row in df.iterrows():
        try:
            print(f"\n🎧 Processing row {idx}")
            yt_links = json.loads(row['YT_link']) if pd.notna(row['YT_link']) else []
            mp3_paths = []
            wav_paths = []

            for link in yt_links:
                if not link or not isinstance(link, str):
                    mp3_paths.append(None)
                    wav_paths.append(None)
                    continue

                print(f"🔗 Downloading: {link}")
                try:
                    mp3_path, safe_title = download_audio_yt_dlp(link, MP3_DIR)
                    wav_path = convert_mp3_to_wav(mp3_path, WAV_DIR, safe_title)

                    # Store relative paths
                    mp3_paths.append(os.path.relpath(mp3_path, start=os.getcwd()))
                    wav_paths.append(os.path.relpath(wav_path, start=os.getcwd()))

                    print(f"   → MP3: {mp3_path}")
                    print(f"   → WAV: {wav_path}")

                except Exception as e:
                    print(f"   ❌ Failed to process {link}: {e}")
                    mp3_paths.append(None)
                    wav_paths.append(None)

            df.at[idx, 'File_mp3'] = json.dumps(mp3_paths)
            df.at[idx, 'File_wav'] = json.dumps(wav_paths)

        except Exception as e:
            print(f"❌ Error in row {idx}: {e}")

    # Save the updated DataFrame
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ All done. Output saved to: {OUTPUT_CSV}")

# === RUN ===
if __name__ == "__main__":
    process_csv()
