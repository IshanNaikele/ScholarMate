import yt_dlp
import os
import re
from groq import Groq
import tempfile
from dotenv import load_dotenv
from typing import Optional

load_dotenv()  # Load environment variables from .env file

def normalize_youtube_url(url: str) -> Optional[str]:
    """
    Normalizes a YouTube video URL. If it's a playlist, extracts the first video.
    """
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
    """
    Extracts the first video URL from a YouTube playlist.
    """
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': 'in_playlist'}) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
            if 'entries' in info and info['entries']:
                first_video_id = info['entries'][0]['id']
                return f"https://www.youtube.com/watch?v={first_video_id}"
    except Exception:
        return None

def extract_audio_from_youtube(youtube_url: str) -> Optional[str]:
    """
    Downloads audio from a YouTube video as a temporary MP3 file.
    """
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

def transcribe_audio_with_groq(audio_file_path: str, language: Optional[str] = None) -> Optional[str]:
    """
    Uses Groq (Whisper-large-v3) to transcribe an audio file.
    If language is None, Whisper auto-detects the language.
    """
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
            return transcription
    except Exception:
        return None

def get_youtube_transcript(youtube_url: str, language: Optional[str] = None) -> str:
    """
    Main function to get the transcript from a YouTube video or playlist URL.
    """
    audio_file = None
    try:
        audio_file = extract_audio_from_youtube(youtube_url)
        if audio_file:
            transcript = transcribe_audio_with_groq(audio_file, language)
            return transcript if transcript else "Could not generate transcript."
        else:
            return "Failed to extract audio or video is unavailable. Please check the URL."
    except Exception:
        return "An unexpected error occurred during processing."
    finally:
        if audio_file and os.path.exists(audio_file):
            try:
                os.remove(audio_file)
            except OSError:
                pass

if __name__ == "__main__":
    print("🔗 Enter a YouTube video or playlist URL:")
    user_input_url = input("> ").strip()

    print("\n🌐 Enter language code (e.g., en for English, hi for Hindi) or press Enter to auto-detect:")
    user_input_lang = input("> ").strip() or None  # Auto-detect if left blank

    if not os.environ.get("GROQ_API_KEY_NEW"):
        print("❌ Error: GROQ_API_KEY_NEW not set. Please check your .env file.")
    else:
        result = get_youtube_transcript(user_input_url, language=user_input_lang)
        print("\n📜 Transcript:\n")
        print(result)
