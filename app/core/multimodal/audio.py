from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional
from app.config import get_settings
from app.utils.logger import log

settings = get_settings()

SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB per chunk


class AudioProcessor:
    """音频处理：语音转写（Whisper API / 本地Whisper）"""

    def __init__(self):
        self._client = None
        self._local_model = None

    def _get_openai_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=settings.EMBEDDING_API_KEY,
                base_url=settings.EMBEDDING_BASE_URL,
            )
        return self._client

    def _get_local_model(self):
        if self._local_model is None:
            import whisper
            self._local_model = whisper.load_model("base")
        return self._local_model

    def transcribe(self, audio_path: str, use_local: bool = False) -> dict:
        """
        转写音频文件
        返回: {"text": str, "segments": list[dict], "language": str}
        """
        path = Path(audio_path)
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {path.suffix}")

        if use_local:
            return self._transcribe_local(audio_path)
        return self._transcribe_api(audio_path)

    def _transcribe_api(self, audio_path: str) -> dict:
        """使用OpenAI Whisper API转写"""
        client = self._get_openai_client()
        file_size = os.path.getsize(audio_path)

        if file_size <= MAX_FILE_SIZE:
            return self._transcribe_single(client, audio_path)

        # 大文件分片处理
        return self._transcribe_chunked(client, audio_path)

    def _transcribe_single(self, client, audio_path: str) -> dict:
        with open(audio_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
            )
        result = resp if isinstance(resp, dict) else resp.__dict__
        text = result.get("text", "")
        segments = result.get("segments", [])
        language = result.get("language", "unknown")

        log.info(f"Transcribed {Path(audio_path).name}: {len(text)} chars, lang={language}")
        return {
            "text": text,
            "segments": [
                {"start": s.get("start", 0), "end": s.get("end", 0), "text": s.get("text", "")}
                for s in segments
            ],
            "language": language,
        }

    def _transcribe_chunked(self, client, audio_path: str) -> dict:
        """大文件分片转写"""
        chunks = self._split_audio(audio_path)
        all_text = []
        all_segments = []
        offset = 0.0

        for chunk_path in chunks:
            try:
                result = self._transcribe_single(client, chunk_path)
                all_text.append(result["text"])
                for seg in result["segments"]:
                    seg["start"] += offset
                    seg["end"] += offset
                    all_segments.append(seg)
                if result["segments"]:
                    offset = all_segments[-1]["end"]
            finally:
                os.unlink(chunk_path)

        return {
            "text": "\n".join(all_text),
            "segments": all_segments,
            "language": all_segments[0].get("language", "unknown") if all_segments else "unknown",
        }

    def _split_audio(self, audio_path: str) -> list[str]:
        """使用ffmpeg将音频分片"""
        import ffmpeg
        duration = self._get_duration(audio_path)
        chunks = []
        start = 0
        chunk_duration = 600  # 10分钟一片

        while start < duration:
            end = min(start + chunk_duration, duration)
            tmp = tempfile.NamedTemporaryFile(suffix=Path(audio_path).suffix, delete=False)
            tmp.close()
            (
                ffmpeg
                .input(audio_path, ss=start, t=end - start)
                .output(tmp.name, acodec="copy")
                .overwrite_output()
                .run(quiet=True)
            )
            chunks.append(tmp.name)
            start = end

        return chunks

    def _get_duration(self, audio_path: str) -> float:
        import ffmpeg
        probe = ffmpeg.probe(audio_path)
        return float(probe["format"]["duration"])

    def _transcribe_local(self, audio_path: str) -> dict:
        """使用本地Whisper模型转写"""
        model = self._get_local_model()
        result = model.transcribe(audio_path)
        segments = result.get("segments", [])
        log.info(f"Local transcribed {Path(audio_path).name}: {len(result['text'])} chars")
        return {
            "text": result["text"],
            "segments": [
                {"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in segments
            ],
            "language": result.get("language", "unknown"),
        }

    def extract_audio_from_video(self, video_path: str) -> str:
        """从视频中提取音频"""
        import ffmpeg
        output = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        output.close()
        (
            ffmpeg
            .input(video_path)
            .output(output.name, acodec="libmp3lame", q=0)
            .overwrite_output()
            .run(quiet=True)
        )
        log.info(f"Extracted audio from {Path(video_path).name}")
        return output.name


_audio_processor: AudioProcessor | None = None


def get_audio_processor() -> AudioProcessor:
    global _audio_processor
    if _audio_processor is None:
        _audio_processor = AudioProcessor()
    return _audio_processor
