# Video Censor Website V1

Features:
- Upload a video
- Automatic common profanity detection
- Custom single words and multi-word phrases
- Mute detected speech with FFmpeg
- Download a censored MP4

Requirements:
- Python 3.10+
- FFmpeg installed

Run:
1. pip install -r requirements.txt
2. python app.py
3. Open http://localhost:5000

The first run downloads the Whisper small model.

For a public production site, long video jobs should eventually use a background queue and persistent storage.
