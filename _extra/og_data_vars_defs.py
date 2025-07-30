import csv

audio_features = [
    {
        "Feature": "Artist",
        "Definition": "Name of the performing artist or band.",
        "Variable Type": "String",
        "Output Type": "Single string",
        "Extractable via Python": "No",
        "Python Library": "N/A",
        "Notes": "Usually human entered or scraped from web or metadata."
    },
    {
        "Feature": "Song Title",
        "Definition": "Name of the song or track title.",
        "Variable Type": "String",
        "Output Type": "Single string",
        "Extractable via Python": "No",
        "Python Library": "N/A",
        "Notes": "Usually human entered or scraped from web or metadata."
    },
    {
        "Feature": "Genre",
        "Definition": "Musical style or category, possibly multiple genres.",
        "Variable Type": "String Array",
        "Output Type": "List of strings",
        "Extractable via Python": "No",
        "Python Library": "N/A",
        "Notes": "Typically human assigned or scraped from databases."
    },
    {
        "Feature": "Mean (of features)",
        "Definition": "Average value over time of any time-varying audio feature.",
        "Variable Type": "Numeric",
        "Output Type": "Single number",
        "Extractable via Python": "Yes",
        "Python Library": "numpy, scipy",
        "Notes": "Apply to any time-series feature (e.g., MFCCs, RMS)."
    },
    {
        "Feature": "Variance",
        "Definition": "Measure of spread or variability of feature over time.",
        "Variable Type": "Numeric",
        "Output Type": "Single number",
        "Extractable via Python": "Yes",
        "Python Library": "numpy, scipy",
        "Notes": "Use numpy.var()"
    },
    {
        "Feature": "Skewness",
        "Definition": "Measure of asymmetry of the feature distribution.",
        "Variable Type": "Numeric",
        "Output Type": "Single number",
        "Extractable via Python": "Yes",
        "Python Library": "scipy.stats",
        "Notes": "Use scipy.stats.skew()"
    },
    {
        "Feature": "Kurtosis",
        "Definition": "Measure of tail heaviness of the feature distribution.",
        "Variable Type": "Numeric",
        "Output Type": "Single number",
        "Extractable via Python": "Yes",
        "Python Library": "scipy.stats",
        "Notes": "Use scipy.stats.kurtosis()"
    },
    {
        "Feature": "Zero Crossing Rate",
        "Definition": "Rate audio waveform crosses zero amplitude axis.",
        "Variable Type": "Numeric",
        "Output Type": "Single number",
        "Extractable via Python": "Yes",
        "Python Library": "librosa",
        "Notes": "librosa.feature.zero_crossing_rate()"
    },
    {
        "Feature": "RMS Energy",
        "Definition": "Root mean square of audio amplitude, approximates loudness.",
        "Variable Type": "Numeric",
        "Output Type": "Single number",
        "Extractable via Python": "Yes",
        "Python Library": "librosa",
        "Notes": "librosa.feature.rms()"
    },
    {
        "Feature": "Loudness",
        "Definition": "Perceived loudness, usually in decibels.",
        "Variable Type": "Numeric",
        "Output Type": "Single number",
        "Extractable via Python": "Yes",
        "Python Library": "librosa, pydub",
        "Notes": "Estimate via RMS or pydub gain"
    },
    {
        "Feature": "Energy",
        "Definition": "Total power of audio signal over time.",
        "Variable Type": "Numeric",
        "Output Type": "Single number",
        "Extractable via Python": "Yes",
        "Python Library": "librosa, numpy",
        "Notes": "Mean RMS squared or signal power"
    },
    {
        "Feature": "Tempo",
        "Definition": "Estimated beats per minute of the track.",
        "Variable Type": "Numeric",
        "Output Type": "Single number",
        "Extractable via Python": "Yes",
        "Python Library": "librosa",
        "Notes": "librosa.beat.beat_track()"
    },
    {
        "Feature": "Time Signature",
        "Definition": "Beats per measure, rhythmic grouping.",
        "Variable Type": "Numeric",
        "Output Type": "Single number",
        "Extractable via Python": "Yes (approximate)",
        "Python Library": "librosa, essentia",
        "Notes": "Estimated from beat/bar detection"
    },
    {
        "Feature": "Danceability",
        "Definition": "How suitable a track is for dancing.",
        "Variable Type": "Numeric",
        "Output Type": "Single number",
        "Extractable via Python": "Approximate",
        "Python Library": "essentia, custom ML",
        "Notes": "Requires ML or Essentia for approximation"
    },
    {
        "Feature": "Speechiness",
        "Definition": "Detects presence of spoken words in audio.",
        "Variable Type": "Numeric",
        "Output Type": "Single number",
        "Extractable via Python": "No",
        "Python Library": "N/A",
        "Notes": "Spotify API or ML only"
    },
    {
        "Feature": "Acousticness",
        "Definition": "Likelihood track is acoustic.",
        "Variable Type": "Numeric",
        "Output Type": "Single number",
        "Extractable via Python": "No",
        "Python Library": "N/A",
        "Notes": "Spotify API or ML only"
    },
    {
        "Feature": "Instrumentalness",
        "Definition": "Probability that track contains no vocals.",
        "Variable Type": "Numeric",
        "Output Type": "Single number",
        "Extractable via Python": "No",
        "Python Library": "N/A",
        "Notes": "Spotify API or ML only"
    },
    {
        "Feature": "Liveness",
        "Definition": "Likelihood audio was recorded live.",
        "Variable Type": "Numeric",
        "Output Type": "Single number",
        "Extractable via Python": "No",
        "Python Library": "N/A",
        "Notes": "Spotify API or ML only"
    },
    {
        "Feature": "Valence",
        "Definition": "Positiveness conveyed by track’s mood.",
        "Variable Type": "Numeric",
        "Output Type": "Single number",
        "Extractable via Python": "No",
        "Python Library": "N/A",
        "Notes": "Requires ML or Spotify data"
    },
    {
        "Feature": "Key / Key Name",
        "Definition": "Musical key of the track (e.g., C, D#, F minor).",
        "Variable Type": "Categorical",
        "Output Type": "Single string",
        "Extractable via Python": "Yes",
        "Python Library": "librosa, essentia",
        "Notes": "Estimated from chroma features"
    },
    {
        "Feature": "Mode / Mode Name",
        "Definition": "Major or minor tonality.",
        "Variable Type": "Categorical",
        "Output Type": "Single string",
        "Extractable via Python": "Yes",
        "Python Library": "librosa, essentia",
        "Notes": "Derived from key detection"
    },
    {
        "Feature": "Explicit (Lyrics)",
        "Definition": "Whether track contains explicit content.",
        "Variable Type": "Categorical",
        "Output Type": "Single value",
        "Extractable via Python": "No",
        "Python Library": "N/A",
        "Notes": "Metadata only"
    },
    {
        "Feature": "FFT (Amplitude vs Frequency)",
        "Definition": "Frequency spectrum magnitude of audio signal.",
        "Variable Type": "Numeric Array",
        "Output Type": "1D Array",
        "Extractable via Python": "Yes",
        "Python Library": "numpy, scipy",
        "Notes": "np.fft.fft() and np.abs()"
    },
    {
        "Feature": "STFT (Short-Time Fourier Transform)",
        "Definition": "Time-varying frequency content representation.",
        "Variable Type": "Complex Array",
        "Output Type": "2D Matrix",
        "Extractable via Python": "Yes",
        "Python Library": "librosa",
        "Notes": "librosa.stft()"
    },
    {
        "Feature": "Mel-Spectrogram",
        "Definition": "Spectrogram with Mel scale frequency bins.",
        "Variable Type": "Numeric Array",
        "Output Type": "2D Matrix",
        "Extractable via Python": "Yes",
        "Python Library": "librosa",
        "Notes": "librosa.feature.melspectrogram()"
    },
    {
        "Feature": "Frequency vs dB Spectrum",
        "Definition": "Frequency spectrum converted to decibel scale.",
        "Variable Type": "Numeric Array",
        "Output Type": "1D or 2D Matrix",
        "Extractable via Python": "Yes",
        "Python Library": "librosa",
        "Notes": "librosa.power_to_db() on FFT or STFT"
    },
    {
        "Feature": "Duration (ms)",
        "Definition": "Total length of the audio file in milliseconds.",
        "Variable Type": "Numeric",
        "Output Type": "Single number",
        "Extractable via Python": "Yes",
        "Python Library": "librosa",
        "Notes": "librosa.get_duration() * 1000"
    }
]

human_entered_features = {"Artist", "Song Title", "Genre"}

def write_custom_filtered_csv(data, filename="audio_features_mk1.csv"):
    filtered = [
        row for row in data
        if (row["Feature"] in human_entered_features) or (row["Extractable via Python"].lower() in ("yes", "approximate"))
    ]
    with open(filename, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=filtered[0].keys())
        writer.writeheader()
        writer.writerows(filtered)
    print(f"✅ CSV including human-entered and extractable features saved as '{filename}'")

if __name__ == "__main__":
    write_custom_filtered_csv(audio_features)
