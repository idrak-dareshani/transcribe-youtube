import os
from pydub import AudioSegment

# === CONFIGURATION ===
AUDIO_FILE = "your_audio.mp3"          # Input audio file
CHUNK_LENGTH_MIN = 30                  # Chunk duration in minutes
OUTPUT_DIR = "chunks"                  # Temporary chunk storage
TRANSCRIBER = "faster-whisper"         # Options: "whisper" or "faster-whisper"
WHISPER_MODEL = "medium"                # tiny, base, small, medium, large
FINAL_TRANSCRIPT = "full_transcript.txt"

# === DEPENDENCIES ===
if TRANSCRIBER == "whisper":
    import whisper
elif TRANSCRIBER == "faster-whisper":
    from faster_whisper import WhisperModel
else:
    raise ValueError("Invalid TRANSCRIBER option. Choose 'whisper' or 'faster-whisper'.")

# === STEP 1: SPLIT AUDIO INTO CHUNKS ===
def split_audio(input_file, chunk_length_min=30):
    print("[INFO] Splitting audio into chunks...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    audio = AudioSegment.from_file(input_file)
    chunk_length_ms = chunk_length_min * 60 * 1000
    total_chunks = len(audio) // chunk_length_ms + 1
    
    chunk_files = []
    for i in range(total_chunks):
        start = i * chunk_length_ms
        end = min((i+1) * chunk_length_ms, len(audio))
        chunk = audio[start:end]
        chunk_file = os.path.join(OUTPUT_DIR, f"chunk_{i:03d}.mp3")
        chunk.export(chunk_file, format="mp3")
        chunk_files.append(chunk_file)
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
