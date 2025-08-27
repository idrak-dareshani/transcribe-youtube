import os
import glob
import shlex
import subprocess

# === CONFIGURATION ===
AUDIO_FILE = "data/Vitamin D Expert： The Fastest Way To Dementia & The Big Lie About Sunlight! [wQJlGHVmdrA].mp3"          # Input audio file
CHUNK_LENGTH_MIN = 10                  # Chunk duration in minutes
OUTPUT_DIR = "chunks"                  # Temporary chunk storage
TRANSCRIBER = "faster-whisper"         # Options: "whisper" or "faster-whisper"
WHISPER_MODEL = "medium"                # tiny, base, small, medium, large
FINAL_TRANSCRIPT = "Vitamin D Expert： The Fastest Way To Dementia & The Big Lie About Sunlight! [wQJlGHVmdrA].txt"

# === DEPENDENCIES ===
if TRANSCRIBER == "whisper":
    import whisper
elif TRANSCRIBER == "faster-whisper":
    from faster_whisper import WhisperModel
else:
    raise ValueError("Invalid TRANSCRIBER option. Choose 'whisper' or 'faster-whisper'.")

# === STEP 1: SPLIT AUDIO INTO CHUNKS ===
def split_audio(input_file, chunk_length_min=30):
    print("[INFO] Splitting audio into chunks using ffmpeg...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Output pattern (e.g., chunk_000.mp3, chunk_001.mp3, etc.)
    output_pattern = os.path.join(OUTPUT_DIR, "chunk_%03d.mp3")

    # FFmpeg command
    cmd = (
        f'ffmpeg -hide_banner -loglevel error -i "{input_file}" '
        f'-f segment -segment_time {chunk_length_min * 60} '
        f'-c copy "{output_pattern}"'
    )

    # Execute
    subprocess.run(shlex.split(cmd), check=True)

    # Get list of created chunk files
    chunk_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "chunk_*.mp3")))

    for chunk_file in chunk_files:
        print(f"  - Saved: {chunk_file}")

    return chunk_files

# === STEP 2: TRANSCRIBE CHUNKS ===
def transcribe_chunks(chunk_files, transcriber="whisper", model_name="small"):
    full_text = ""

    if transcriber == "whisper":
        print(f"[INFO] Loading openai-whisper model: {model_name}")
        model = whisper.load_model(model_name)
        
        for i, chunk_path in enumerate(chunk_files):
            print(f"[INFO] Transcribing chunk {i+1}/{len(chunk_files)} with whisper: {chunk_path}")
            result = model.transcribe(chunk_path, language="English")
            full_text += f"\n\n--- Chunk {i+1} ---\n\n"
            full_text += result["text"]

    elif transcriber == "faster-whisper":
        print(f"[INFO] Loading faster-whisper model: {model_name}")
        model = WhisperModel(model_name, compute_type="int8")  # or "float16" if GPU is available
        
        for i, chunk_path in enumerate(chunk_files):
            print(f"[INFO] Transcribing chunk {i+1}/{len(chunk_files)} with faster-whisper: {chunk_path}")
            segments, info = model.transcribe(chunk_path, language="en")
            full_text += f"\n\n--- Chunk {i+1} ---\n\n"
            for segment in segments:
                full_text += segment.text + " "

    else:
        raise ValueError("Unsupported transcriber. Choose 'whisper' or 'faster-whisper'.")

    return full_text.strip()

# === STEP 3: SAVE TRANSCRIPT ===
def save_transcript(transcript, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"[INFO] Transcript saved to: {output_file}")

# === MAIN FUNCTION ===
def main():
    chunks = split_audio(AUDIO_FILE, CHUNK_LENGTH_MIN)
    transcript = transcribe_chunks(chunks, TRANSCRIBER, WHISPER_MODEL)
    save_transcript(transcript, FINAL_TRANSCRIPT)

if __name__ == "__main__":
    main()
