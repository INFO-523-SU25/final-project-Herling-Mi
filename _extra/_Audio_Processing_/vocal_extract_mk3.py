import os
import subprocess
import concurrent.futures
from datetime import datetime

# Constants
OUTPUT_FOLDER = "demucs_output"
LOG_FOLDER = "demucs_logs"
MISSING_LOG = os.path.join(LOG_FOLDER, "missing_files.txt")
ERROR_LOG = os.path.join(LOG_FOLDER, "error_files.txt")
MAX_THREADS = 10  # Limit number of concurrent threads

# Ensure folders exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# Thread-safe logging
def log_message(log_path, message):
    with open(log_path, "a", encoding="utf-8") as f:
        timestamp = datetime.now().isoformat()
        f.write(f"{timestamp} - {message}\n")

def process_audio_file(audio_file):
    if not os.path.exists(audio_file):
        msg = f"❌ File not found: {audio_file}"
        print(msg)
        log_message(MISSING_LOG, msg)
        return

    command = [
        "demucs",
        "--two-stems=vocals",
        "--out", OUTPUT_FOLDER,
        audio_file
    ]

    print(f"🔄 Running Demucs on: {audio_file}")
    try:
        subprocess.run(command, check=True)
        print(f"✅ Separation complete for {audio_file}")
    except subprocess.CalledProcessError as e:
        msg = f"❌ Error processing {audio_file}: {e}"
        print(msg)
        log_message(ERROR_LOG, msg)

def separate_audio_files_thread_pool(audio_files):
    total_files = len(audio_files)
    print(f"\n🧵 Launching up to {MAX_THREADS} threads to process {total_files} audio file(s)...\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(process_audio_file, file) for file in audio_files]

        for future in concurrent.futures.as_completed(futures):
            future.result()  # Triggers exception handling if any error occurred

    print("\n🎉 All audio files processed.\n")

if __name__ == "__main__":
    # Example list of audio files
    audio_files = [
        r"C:\Users\Mr_Green\OneDrive\Desktop\0_Sum_25\INFO_523\HWs\group_project\audio_wavs\Gloria_Gaynor_I_Will_Survive.wav",
        r"C:\Users\Mr_Green\OneDrive\Desktop\0_Sum_25\INFO_523\HWs\group_project\Yashis_Suite\Final_org_attempt_mk1\Yashis_processed_music_list\wav\[MV] MeloMance(멜로망스) _ Love, Maybe(사랑인가 봐) (사내맞선 OST 스페셜 트랙) (Live Clip Ver.).wav",
        # Add more paths here
    ]

    separate_audio_files_thread_pool(audio_files)
