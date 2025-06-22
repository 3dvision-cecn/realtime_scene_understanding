import ffmpeg
import numpy as np
import cv2


class VideoLoader:
    def __init__(self, video_path: str, max_length: int = -1):
        self.video_path = video_path
        self.max_length = max_length

        # ── probe stream ───────────────────────────────────────────────
        probe = ffmpeg.probe(video_path)
        v_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")

        self.width  = int(v_stream["width"])
        self.height = int(v_stream["height"])

        # average frame-rate comes as "num/den" (e.g. "30000/1001")
        num, den   = map(int, v_stream["avg_frame_rate"].split("/"))
        self.fps   = num / den if den else 0.0           # guard ÷0 for weird files
        self._dt   = 1.0 / self.fps if self.fps else 0.0
        self._idx  = 0                                    # next frame index

        self._frame_bytes = self.width * self.height * 3  # BGR24

        # ── persistent ffmpeg process ─────────────────────────────────
        self._proc = (
            ffmpeg
            .input(video_path)
            .output("pipe:", format="rawvideo", pix_fmt="bgr24")
            .run_async(pipe_stdout=True, pipe_stderr=True, quiet=True)
        )

    # ────────────────────────────────────────────────────────────────────
    def next_frame(self):
        """
        Returns
        -------
        (np.ndarray, float) | None
            frame      – shape (H, W, 3), dtype=uint8, RGB
            timestamp  – seconds from start of clip (float)
            None       – when video is exhausted
        """
        raw = self._proc.stdout.read(self._frame_bytes)
        if len(raw) < self._frame_bytes:          # EOF
            self._proc.stdout.close()
            self._proc.wait()
            return None

        # decode & color-convert
        bgr   = np.frombuffer(raw, np.uint8).reshape(self.height, self.width, 3)
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # timestamp of this frame
        timestamp = self._idx * self._dt
        self._idx += 1

        if self._idx >= self.max_length and self.max_length > 0:
            self._proc.stdout.close()
            self._proc.wait()
            return None

        return frame, timestamp
