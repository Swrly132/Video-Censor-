import os, re, uuid, subprocess, gc
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from faster_whisper import WhisperModel

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

DEFAULT_BLOCKED_WORDS = {
    "fuck", "fucking", "shit", "bitch", "asshole",
    "damn", "cunt", "bastard", "dick", "piss"
}


def norm(text):
    return re.sub(r"[^a-z0-9']+", "", text.lower()).strip()

def build_intervals(words, blocked_terms):
    phrases = []
    for term in blocked_terms:
        toks = [norm(t) for t in term.split()]
        toks = [t for t in toks if t]
        if toks:
            phrases.append(toks)

    normalized = [norm(w["word"]) for w in words]
    intervals = []
    for i in range(len(words)):
        for phrase in phrases:
            n = len(phrase)
            if i + n <= len(words) and normalized[i:i+n] == phrase:
                intervals.append((
                    max(0, words[i]["start"] - 0.04),
                    words[i+n-1]["end"] + 0.04
                ))

    intervals.sort()
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + 0.03:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(s, e) for s, e in merged]

def transcribe_words(video_path):
    audio_path = video_path.with_suffix(".wav")

    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(audio_path)
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    model = WhisperModel(
        "tiny.en",
        device="cpu",
        compute_type="int8",
        cpu_threads=1,
        num_workers=1
    )

    try:
        segments, _ = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=True,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False
        )

        words = []

        for seg in segments:
            for w in (seg.words or []):
                words.append({
                    "word": w.word,
                    "start": float(w.start),
                    "end": float(w.end)
                })

        return words

    finally:
        del model
        gc.collect()
        audio_path.unlink(missing_ok=True)

def censor_video(input_path, output_path, intervals):
    if not intervals:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path), "-c", "copy", str(output_path)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return

    enable_expr = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in intervals)
    audio_filter = f"volume=enable='{enable_expr}':volume=0"

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path), "-c:v", "copy",
         "-af", audio_filter, "-c:a", "aac", "-b:a", "192k", str(output_path)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

@app.get("/")
def index():
    return app.send_static_file("index.html")

@app.post("/api/censor")
def censor():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded."}), 400

    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "No file selected."}), 400

    custom_terms = {
        line.strip().lower()
        for line in request.form.get("custom_words", "").splitlines()
        if line.strip()
    }
    blocked = set(custom_terms)
    if request.form.get("auto_profanity", "true").lower() == "true":
        blocked |= DEFAULT_BLOCKED_WORDS

    job = uuid.uuid4().hex
    suffix = Path(f.filename).suffix or ".mp4"
    input_path = UPLOAD_DIR / f"{job}{suffix}"
    output_name = f"{job}_censored.mp4"
    output_path = OUTPUT_DIR / output_name
    f.save(input_path)

    try:
        words = transcribe_words(input_path)
        intervals = build_intervals(words, blocked)
        censor_video(input_path, output_path, intervals)
    except subprocess.CalledProcessError as e:
        return jsonify({
            "error": "FFmpeg failed. Make sure FFmpeg is installed.",
            "details": e.stderr.decode("utf-8", errors="ignore")[-1000:]
        }), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        input_path.unlink(missing_ok=True)

    return jsonify({
        "download_url": f"/download/{output_name}",
        "censored_sections": len(intervals)
    })

@app.get("/download/<filename>")
def download(filename):
    return send_from_directory(
        OUTPUT_DIR, filename, as_attachment=True,
        download_name="censored_video.mp4"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
