import subprocess, shlex

def split_audio_ffmpeg(audio_path, segment_seconds=600):
    cmd = f'ffmpeg -i "{audio_path}" -f segment -segment_time {segment_seconds} -c copy chunk_%03d.mp3'
    subprocess.run(shlex.split(cmd), check=True)

if __name__ == "__main__":
    split_audio_ffmpeg("Vitamin D Expert： The Fastest Way To Dementia & The Big Lie About Sunlight! [wQJlGHVmdrA].mp3", 
                       segment_seconds=1800)

    # proceed to transcribe with your preferred method

