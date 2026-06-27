from __future__ import annotations

import os
import tempfile
from pathlib import Path
from app.utils.logger import log

SUPPORTED_FORMATS = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm"}


class VideoProcessor:
    """视频处理：关键帧提取 + 音频转写 + 内容分析"""

    def __init__(self):
        self._image_processor = None
        self._audio_processor = None

    def _get_image_processor(self):
        if self._image_processor is None:
            from app.core.multimodal.image import get_image_processor
            self._image_processor = get_image_processor()
        return self._image_processor

    def _get_audio_processor(self):
        if self._audio_processor is None:
            from app.core.multimodal.audio import get_audio_processor
            self._audio_processor = get_audio_processor()
        return self._audio_processor

    def extract_frames(self, video_path: str, interval: int = 30, max_frames: int = 20) -> list[str]:
        """按时间间隔提取关键帧"""
        import ffmpeg
        path = Path(video_path)
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported video format: {path.suffix}")

        duration = self._get_duration(video_path)
        output_dir = tempfile.mkdtemp()
        frames = []
        timestamp = 0
        count = 0

        while timestamp < duration and count < max_frames:
            output_path = os.path.join(output_dir, f"frame_{count:04d}.jpg")
            try:
                (
                    ffmpeg
                    .input(video_path, ss=timestamp)
                    .output(output_path, vframes=1, format="image2", vcodec="mjpeg")
                    .overwrite_output()
                    .run(quiet=True)
                )
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    frames.append(output_path)
                    count += 1
            except Exception as e:
                log.warning(f"Failed to extract frame at {timestamp}s: {e}")
            timestamp += interval

        log.info(f"Extracted {len(frames)} frames from {path.name} (interval={interval}s)")
        return frames

    def process(self, video_path: str, interval: int = 30) -> dict:
        """
        完整视频处理流程
        返回: {"frames": [...], "transcript": str, "combined": str, "frame_descriptions": [...]}
        """
        path = Path(video_path)
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported video format: {path.suffix}")

        result = {
            "frames": [],
            "frame_descriptions": [],
            "transcript": "",
            "combined": "",
        }

        # 1. 提取关键帧并分析
        frames = self.extract_frames(video_path, interval=interval)
        result["frames"] = frames

        img_proc = self._get_image_processor()
        frame_texts = []
        for i, frame_path in enumerate(frames):
            try:
                frame_result = img_proc.process(frame_path)
                result["frame_descriptions"].append({
                    "frame_index": i,
                    "timestamp": i * interval,
                    "ocr_text": frame_result["ocr_text"],
                    "description": frame_result["description"],
                })
                if frame_result["ocr_text"]:
                    frame_texts.append(f"[帧{i} @ {i*interval}s] {frame_result['ocr_text']}")
            except Exception as e:
                log.warning(f"Failed to process frame {i}: {e}")

        # 2. 提取音频并转写
        audio_path = None
        try:
            audio_proc = self._get_audio_processor()
            audio_path = audio_proc.extract_audio_from_video(video_path)
            audio_result = audio_proc.transcribe(audio_path)
            result["transcript"] = audio_result["text"]
        except Exception as e:
            log.warning(f"Audio extraction/transcription failed: {e}")
        finally:
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)

        # 3. 合并所有结果
        parts = []
        parts.append(f"[视频文件: {path.name}, 时长: {self._format_duration(self._get_duration(video_path))}]")
        if frame_texts:
            parts.append(f"[关键帧内容]\n" + "\n".join(frame_texts))
        if result["transcript"]:
            parts.append(f"[音频转写]\n{result['transcript']}")

        result["combined"] = "\n\n".join(parts)
        log.info(f"Processed video {path.name}: {len(frames)} frames, transcript={len(result['transcript'])} chars")

        # 清理临时帧文件
        for frame in frames:
            try:
                os.unlink(frame)
                os.rmdir(os.path.dirname(frame))
            except Exception:
                pass

        return result

    def _get_duration(self, video_path: str) -> float:
        import ffmpeg
        probe = ffmpeg.probe(video_path)
        return float(probe["format"]["duration"])

    def _format_duration(self, seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"


_video_processor: VideoProcessor | None = None


def get_video_processor() -> VideoProcessor:
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor()
    return _video_processor
