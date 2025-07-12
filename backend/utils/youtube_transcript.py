import yt_dlp
import os
import re
from groq import Groq
import tempfile
from dotenv import load_dotenv
from typing import Optional
from pydub import AudioSegment
import math

load_dotenv()  # Load environment variables from .env file

def normalize_youtube_url(url: str) -> Optional[str]:
    if "playlist?list=" in url:
        return get_first_video_from_playlist(url)

    regex_patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([\w-]{11})",
        r"(?:https?://)?youtu\.be/([\w-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/(?:v/|e/)([\w-]{11})",
        r"http://googleusercontent\.com/youtube\.com/([\w-]{11})"
    ]

    for pattern in regex_patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            return f"https://www.youtube.com/watch?v={video_id}"
    return None

def get_first_video_from_playlist(playlist_url: str) -> Optional[str]:
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': 'in_playlist'}) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
            if 'entries' in info and info['entries']:
                first_video_id = info['entries'][0]['id']
                return f"https://www.youtube.com/watch?v={first_video_id}"
    except Exception:
        return None

def extract_audio_from_youtube(youtube_url: str) -> Optional[str]:
    normalized_url = normalize_youtube_url(youtube_url)
    if not normalized_url:
        return None

    try:
        temp_dir = tempfile.gettempdir()
        temp_audio_path = os.path.join(temp_dir, "yt_audio.%(ext)s")
        final_audio_path = temp_audio_path.replace("%(ext)s", "mp3")

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': temp_audio_path,
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([normalized_url])

        return final_audio_path if os.path.exists(final_audio_path) else None
    except Exception:
        return None

def split_audio_into_chunks(audio_file_path: str, max_size_mb: int = 25) -> list[str]:
    try:
        audio = AudioSegment.from_mp3(audio_file_path)
        file_size_bytes = os.path.getsize(audio_file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)

        if file_size_mb <= max_size_mb:
            return [audio_file_path]

        total_length_ms = len(audio)
        num_chunks = math.ceil(file_size_mb / max_size_mb)
        chunk_length_ms = total_length_ms / num_chunks

        chunk_paths = []
        temp_dir = tempfile.gettempdir()

        for i in range(num_chunks):
            start_ms = int(i * chunk_length_ms)
            end_ms = int(min((i + 1) * chunk_length_ms, total_length_ms))
            chunk = audio[start_ms:end_ms]
            chunk_filename = os.path.join(temp_dir, f"yt_audio_chunk_{i}.mp3")
            chunk.export(chunk_filename, format="mp3")
            chunk_paths.append(chunk_filename)
        return chunk_paths
    except Exception as e:
        print(f"Error splitting audio: {e}")
        return []

def transcribe_audio_with_groq(audio_file_path: str, language: Optional[str] = None) -> Optional[str]:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY_NEW"))

    if not os.path.exists(audio_file_path):
        return None

    try:
        with open(audio_file_path, "rb") as file:
            transcription_params = {
                "file": file,
                "model": "whisper-large-v3",
                "response_format": "text"
            }
            if language:
                transcription_params["language"] = language

            transcription = client.audio.transcriptions.create(**transcription_params)
            return transcription  # fixed: already string
    except Exception as e:
        print(f"Error transcribing chunk {audio_file_path}: {e}")
        return None

def get_youtube_transcript(youtube_url: str, language: Optional[str] = None) -> str:
    full_transcript_parts = []
    downloaded_audio_path = None
    chunk_files = []

    try:
        downloaded_audio_path = extract_audio_from_youtube(youtube_url)
        if not downloaded_audio_path:
            return "Failed to extract audio or video is unavailable. Please check the URL."

        chunk_files = split_audio_into_chunks(downloaded_audio_path, max_size_mb=25)

        if not chunk_files:
            return "Failed to split audio into manageable chunks."

        for chunk_file in chunk_files:
            transcript_part = transcribe_audio_with_groq(chunk_file, language)
            if transcript_part:
                full_transcript_parts.append(transcript_part)
            else:
                print(f"Warning: Could not transcribe chunk: {chunk_file}")

        if full_transcript_parts:
            return " ".join(full_transcript_parts)
        else:
            return "Could not generate transcript from any audio chunks."

    except Exception as e:
        return f"An unexpected error occurred during processing: {e}"
    finally:
        if downloaded_audio_path and os.path.exists(downloaded_audio_path):
            try:
                os.remove(downloaded_audio_path)
            except OSError:
                pass
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                try:
                    os.remove(chunk_file)
                except OSError:
                    pass

if __name__ == "__main__":
    print("\U0001F517 Enter a YouTube video or playlist URL:")
    user_input_url = input("> ").strip()

    print("\n\U0001F310 Enter language code (e.g., en for English, hi for Hindi) or press Enter to auto-detect:")
    user_input_lang = input("> ").strip() or None

    if not os.environ.get("GROQ_API_KEY_NEW"):
        print("\u274C Error: GROQ_API_KEY_NEW not set. Please check your .env file.")
    else:
        result = get_youtube_transcript(user_input_url, language=user_input_lang)
        print("\n\U0001F4DC Transcript:\n")
        print(result)
