"""Webcam replay: stream a video file as base64 JPEG frames.

Needs OpenCV and Pillow: ``pip install thalamus[video]``.

Note that a frame is two orders of magnitude larger than an EEG sample, so a video
stream will be the first thing to overrun a slow client's queue. That is realistic
and worth knowing about — but if you only need the *timing* of a webcam and not
its pixels, use a :class:`~thalamus.devices.synthetic.SyntheticDevice` at 30 Hz
instead and save yourself the bandwidth.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

from .base import Reading, RecordingDevice

logger = logging.getLogger(__name__)


class VideoDevice(RecordingDevice):
    """Replay a video file frame by frame, as a camera would.

    The frame rate is read from the file unless you override it with ``rate``.
    """

    def __init__(
        self,
        device_id: str,
        path: Union[str, Path],
        *,
        quality: int = 80,
        max_width: Optional[int] = None,
        loop: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(device_id, **kwargs)
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"[{device_id}] no such video: {self.path}")

        self.quality = quality
        self.max_width = max_width
        self.loop = loop
        self._cv2, self._Image = _require_video()

        if self.rate is None:
            capture = self._cv2.VideoCapture(str(self.path))
            fps = capture.get(self._cv2.CAP_PROP_FPS)
            capture.release()
            self.rate = float(fps) if fps and fps > 0 else 30.0
            logger.info("[%s] using the file's frame rate: %.1f fps", device_id, self.rate)

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.setdefault("channels", ["image"])
        info.setdefault("encoding", "jpeg;base64")
        info.setdefault("source", str(self.path))
        return info

    def samples(self) -> Iterator[Reading]:
        while True:
            # Open once and read sequentially. Reopening the file per frame
            # constructed a fresh VideoCapture and seeked to the target frame for
            # *every single frame*, which turns a linear read into a quadratic one
            # and cannot sustain 30 fps on any real video.
            capture = self._cv2.VideoCapture(str(self.path))
            if not capture.isOpened():
                raise OSError(f"[{self.device_id}] cannot open video: {self.path}")

            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    yield {"image": self._encode(frame)}
            finally:
                capture.release()

            if not self.loop:
                return
            logger.info("[%s] end of %s; looping", self.device_id, self.path.name)

    def _encode(self, frame) -> str:
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        image = self._Image.fromarray(rgb)

        if self.max_width and image.width > self.max_width:
            height = round(image.height * self.max_width / image.width)
            image = image.resize((self.max_width, height))

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def _require_video():
    try:
        import cv2
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "video replay needs OpenCV and Pillow. Install them with: pip install 'thalamus[video]'"
        ) from exc
    return cv2, Image
