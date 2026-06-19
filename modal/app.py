"""Modal deployment for NVIDIA LocateAnything-3B."""

from __future__ import annotations

import base64
import io
import os
import re
import tempfile
from pathlib import Path
from typing import Literal, Optional

import modal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.requests import Request

MODEL_ID = "nvidia/LocateAnything-3B"
MODEL_REVISION = "main"
VOLUME_NAME = "locateanything-weights"
MODEL_DIR = Path("/models") / MODEL_ID
# MoonViT attention memory scales with patch count; keep patches near processor limit.
VISION_TOKEN_LIMIT = 4096
PATCH_SIZE = 14
MAX_IMAGE_LONG_EDGE = int(PATCH_SIZE * (VISION_TOKEN_LIMIT**0.5))  # ~896px square
DEFAULT_VIDEO_FPS = 2.0
DEFAULT_MAX_FRAMES = 16
VIDEO_MAX_PIXELS = VISION_TOKEN_LIMIT * PATCH_SIZE * PATCH_SIZE

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "VIDEO_MAX_PIXELS": str(VIDEO_MAX_PIXELS),
        }
    )
    .pip_install(
        "torch",
        "torchvision",
        "transformers==4.57.1",
        "opencv-python-headless==4.11.0.86",
        "numpy==1.25.0",
        "Pillow==11.1.0",
        "peft",
        "decord==0.6.0",
        "lmdb==1.7.5",
        "fastapi[standard]",
        "huggingface_hub",
        "requests",
    )
)

app = modal.App("locateanything-3b")


class InferenceRequest(BaseModel):
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded image bytes (with or without data URL prefix).",
    )
    image_url: Optional[str] = Field(
        default=None,
        description="Public URL to fetch an image from.",
    )
    generation_mode: Literal["fast", "slow", "hybrid"] = "hybrid"
    max_new_tokens: int = Field(default=2048, ge=1, le=8192)


class DetectRequest(InferenceRequest):
    categories: list[str] = Field(min_length=1)


class GroundRequest(InferenceRequest):
    phrase: str = Field(min_length=1)
    mode: Literal["single", "multi"] = "multi"


class PointRequest(InferenceRequest):
    phrase: str = Field(min_length=1)


class GuiRequest(InferenceRequest):
    phrase: str = Field(min_length=1)
    output_type: Literal["box", "point"] = "box"


class BoxResult(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class PointResult(BaseModel):
    x: float
    y: float


class InferenceResponse(BaseModel):
    answer: str
    boxes: list[BoxResult] = Field(default_factory=list)
    points: list[PointResult] = Field(default_factory=list)
    image_size: list[int]
    latency_ms: float


class VideoInferenceRequest(BaseModel):
    video_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded video bytes (with or without data URL prefix).",
    )
    video_url: Optional[str] = Field(
        default=None,
        description="Public URL to fetch a video from.",
    )
    fps: float = Field(default=DEFAULT_VIDEO_FPS, ge=0.1, le=10.0)
    max_frames: int = Field(default=DEFAULT_MAX_FRAMES, ge=1, le=32)
    video_start: float = Field(default=0.0, ge=0.0)
    video_end: Optional[float] = Field(default=None, ge=0.0)
    generation_mode: Literal["fast", "slow", "hybrid"] = "hybrid"
    max_new_tokens: int = Field(default=4096, ge=1, le=8192)


class VideoDetectRequest(VideoInferenceRequest):
    categories: list[str] = Field(min_length=1)


class VideoGroundRequest(VideoInferenceRequest):
    phrase: str = Field(min_length=1)
    mode: Literal["single", "multi"] = "multi"


class VideoPointRequest(VideoInferenceRequest):
    phrase: str = Field(min_length=1)


class VideoGuiRequest(VideoInferenceRequest):
    phrase: str = Field(min_length=1)
    output_type: Literal["box", "point"] = "box"


class FrameResult(BaseModel):
    frame_index: int
    timestamp_sec: Optional[float] = None
    boxes: list[BoxResult] = Field(default_factory=list)
    points: list[PointResult] = Field(default_factory=list)


class VideoInferenceResponse(BaseModel):
    answer: str
    frames: list[FrameResult] = Field(default_factory=list)
    video_size: list[int]
    frame_count: int
    sampled_fps: float
    latency_ms: float


def _decode_image_base64(image_base64: str):
    from PIL import Image

    payload = image_base64
    if "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(payload)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}") from exc


def _load_image(image_base64: Optional[str], image_url: Optional[str]):
    from PIL import Image
    import requests

    if image_base64:
        return _decode_image_base64(image_base64)
    if image_url:
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image URL: {exc}") from exc
    raise HTTPException(status_code=400, detail="Provide either image_base64 or image_url.")


def _load_video_to_temp(video_base64: Optional[str], video_url: Optional[str]) -> str:
    import requests

    if not video_base64 and not video_url:
        raise HTTPException(
            status_code=400,
            detail="Provide either video_base64 or video_url.",
        )

    suffix = ".mp4"
    if video_url:
        lower_url = video_url.lower()
        for ext in (".mp4", ".webm", ".mov", ".avi", ".mkv"):
            if lower_url.endswith(ext):
                suffix = ext
                break

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            if video_base64:
                payload = video_base64
                if "," in payload:
                    payload = payload.split(",", 1)[1]
                temp_file.write(base64.b64decode(payload))
            else:
                response = requests.get(video_url, timeout=120)
                response.raise_for_status()
                temp_file.write(response.content)
            return temp_file.name
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid video input: {exc}") from exc


def _get_video_size(video_path: str) -> tuple[int, int]:
    import decord

    reader = decord.VideoReader(video_path)
    frame = reader[0].asnumpy()
    height, width = frame.shape[:2]
    return width, height


def _estimate_patch_count(width: int, height: int) -> int:
    return (width // PATCH_SIZE) * (height // PATCH_SIZE)


def _prepare_image_for_inference(image):
    """Downscale images so ViT patch count stays within the GPU-safe token limit."""
    from PIL import Image

    original_width, original_height = image.size
    working = image
    width, height = original_width, original_height

    long_edge = max(width, height)
    if long_edge > MAX_IMAGE_LONG_EDGE:
        scale = MAX_IMAGE_LONG_EDGE / long_edge
        width = max(PATCH_SIZE, int(width * scale))
        height = max(PATCH_SIZE, int(height * scale))
        working = working.resize((width, height), Image.Resampling.LANCZOS)

    patch_count = _estimate_patch_count(width, height)
    if patch_count > VISION_TOKEN_LIMIT:
        scale = (VISION_TOKEN_LIMIT / patch_count) ** 0.5
        width = max(PATCH_SIZE, int(width * scale))
        height = max(PATCH_SIZE, int(height * scale))
        working = working.resize((width, height), Image.Resampling.LANCZOS)

    return working, original_width, original_height


def _scale_boxes_to_original(
    boxes: list[BoxResult],
    inference_width: int,
    inference_height: int,
    original_width: int,
    original_height: int,
) -> list[BoxResult]:
    if inference_width == original_width and inference_height == original_height:
        return boxes

    scale_x = original_width / inference_width
    scale_y = original_height / inference_height
    return [
        BoxResult(
            x1=box.x1 * scale_x,
            y1=box.y1 * scale_y,
            x2=box.x2 * scale_x,
            y2=box.y2 * scale_y,
        )
        for box in boxes
    ]


def _scale_points_to_original(
    points: list[PointResult],
    inference_width: int,
    inference_height: int,
    original_width: int,
    original_height: int,
) -> list[PointResult]:
    if inference_width == original_width and inference_height == original_height:
        return points

    scale_x = original_width / inference_width
    scale_y = original_height / inference_height
    return [
        PointResult(
            x=point.x * scale_x,
            y=point.y * scale_y,
        )
        for point in points
    ]


def _parse_boxes(answer: str, image_width: int, image_height: int) -> list[BoxResult]:
    boxes: list[BoxResult] = []
    for match in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        x1, y1, x2, y2 = [int(group) for group in match.groups()]
        boxes.append(
            BoxResult(
                x1=x1 / 1000 * image_width,
                y1=y1 / 1000 * image_height,
                x2=x2 / 1000 * image_width,
                y2=y2 / 1000 * image_height,
            )
        )
    return boxes


def _parse_points(answer: str, image_width: int, image_height: int) -> list[PointResult]:
    points: list[PointResult] = []
    for match in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
        x, y = int(match.group(1)), int(match.group(2))
        points.append(
            PointResult(
                x=x / 1000 * image_width,
                y=y / 1000 * image_height,
            )
        )
    return points


def _parse_video_frames(
    answer: str,
    width: int,
    height: int,
    include_boxes: bool,
    include_points: bool,
) -> list[FrameResult]:
    frames: list[FrameResult] = []
    segments = re.split(r"(?=Frame-\d+)", answer)
    for segment in segments:
        if not segment.startswith("Frame-"):
            continue
        header_match = re.match(r"Frame-(\d+)(?:-([\d.]+)s)?:", segment)
        if not header_match:
            continue
        frame_index = int(header_match.group(1))
        timestamp_sec = (
            float(header_match.group(2)) if header_match.group(2) else None
        )
        body = segment[header_match.end() :]
        frames.append(
            FrameResult(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                boxes=_parse_boxes(body, width, height) if include_boxes else [],
                points=_parse_points(body, width, height) if include_points else [],
            )
        )

    if frames:
        return frames

    if include_boxes or include_points:
        frames.append(
            FrameResult(
                frame_index=1,
                timestamp_sec=0.0,
                boxes=_parse_boxes(answer, width, height) if include_boxes else [],
                points=_parse_points(answer, width, height) if include_points else [],
            )
        )
    return frames


@app.cls(
    gpu="L40S",
    image=inference_image,
    volumes={MODEL_DIR.as_posix(): volume},
    scaledown_window=300,
    timeout=600,
)
@modal.concurrent(max_inputs=1)
class LocateAnythingModel:
    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
            volume.reload()
            if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
                raise RuntimeError(
                    f"Model weights missing at {MODEL_DIR}. "
                    "Run: python -m modal run modal/download.py::download_model"
                )

        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
        )
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.in_token_limit = VISION_TOKEN_LIMIT
        self.model = AutoModel.from_pretrained(
            MODEL_DIR,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device).eval()

    def _predict(
        self,
        image,
        question: str,
        generation_mode: str,
        max_new_tokens: int,
    ) -> dict:
        import time
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.py_apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            return_tensors="pt",
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        torch.cuda.empty_cache()

        start = time.perf_counter()
        with torch.no_grad():
            response = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"],
                image_grid_hws=image_grid_hws,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode=generation_mode,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                verbose=False,
            )
        latency_ms = (time.perf_counter() - start) * 1000

        answer = response[0] if isinstance(response, tuple) else response
        return {"answer": answer, "latency_ms": latency_ms}

    def _predict_video(
        self,
        video_path: str,
        question: str,
        fps: float,
        max_frames: int,
        video_start: float,
        video_end: Optional[float],
        generation_mode: str,
        max_new_tokens: int,
    ) -> dict:
        import time
        import torch

        video_content: dict = {
            "type": "video",
            "video": video_path,
            "fps": fps,
            "max_frames": max_frames,
            "video_start": video_start,
        }
        if video_end is not None:
            video_content["video_end"] = video_end

        messages = [
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.py_apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos, video_kwargs = self.processor.process_vision_info(
            messages,
            return_video_kwargs=True,
            video_reader_backend="decord",
        )
        processor_kwargs = {"return_tensors": "pt"}
        if video_kwargs:
            processor_kwargs.update(video_kwargs)

        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            **processor_kwargs,
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        torch.cuda.empty_cache()

        start = time.perf_counter()
        with torch.no_grad():
            response = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"],
                image_grid_hws=image_grid_hws,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode=generation_mode,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                verbose=False,
            )
        latency_ms = (time.perf_counter() - start) * 1000

        answer = response[0] if isinstance(response, tuple) else response
        sampled_fps = (
            float(video_kwargs["fps"][0])
            if video_kwargs and video_kwargs.get("fps")
            else fps
        )
        return {
            "answer": answer,
            "latency_ms": latency_ms,
            "sampled_fps": sampled_fps,
        }

    def _run_inference(
        self,
        request: InferenceRequest,
        question: str,
        include_boxes: bool = True,
        include_points: bool = False,
    ) -> InferenceResponse:
        image = _load_image(request.image_base64, request.image_url)
        inference_image, original_width, original_height = _prepare_image_for_inference(
            image
        )
        inference_width, inference_height = inference_image.size
        result = self._predict(
            image=inference_image,
            question=question,
            generation_mode=request.generation_mode,
            max_new_tokens=request.max_new_tokens,
        )
        answer = result["answer"]
        boxes = (
            _scale_boxes_to_original(
                _parse_boxes(answer, inference_width, inference_height),
                inference_width,
                inference_height,
                original_width,
                original_height,
            )
            if include_boxes
            else []
        )
        points = (
            _scale_points_to_original(
                _parse_points(answer, inference_width, inference_height),
                inference_width,
                inference_height,
                original_width,
                original_height,
            )
            if include_points
            else []
        )
        return InferenceResponse(
            answer=answer,
            boxes=boxes,
            points=points,
            image_size=[original_width, original_height],
            latency_ms=result["latency_ms"],
        )

    def _run_video_inference(
        self,
        request: VideoInferenceRequest,
        question: str,
        include_boxes: bool = True,
        include_points: bool = False,
    ) -> VideoInferenceResponse:
        video_path = _load_video_to_temp(request.video_base64, request.video_url)
        try:
            width, height = _get_video_size(video_path)
            result = self._predict_video(
                video_path=video_path,
                question=question,
                fps=request.fps,
                max_frames=request.max_frames,
                video_start=request.video_start,
                video_end=request.video_end,
                generation_mode=request.generation_mode,
                max_new_tokens=request.max_new_tokens,
            )
            frames = _parse_video_frames(
                result["answer"],
                width,
                height,
                include_boxes=include_boxes,
                include_points=include_points,
            )
            return VideoInferenceResponse(
                answer=result["answer"],
                frames=frames,
                video_size=[width, height],
                frame_count=len(frames),
                sampled_fps=result["sampled_fps"],
                latency_ms=result["latency_ms"],
            )
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)

    @modal.method()
    def detect(self, request: DetectRequest) -> InferenceResponse:
        categories = "</c>".join(request.categories)
        question = (
            "Locate all the instances that matches the following description: "
            f"{categories}."
        )
        return self._run_inference(request, question, include_boxes=True)

    @modal.method()
    def ground(self, request: GroundRequest) -> InferenceResponse:
        if request.mode == "single":
            question = (
                "Locate a single instance that matches the following description: "
                f"{request.phrase}."
            )
        else:
            question = (
                "Locate all the instances that match the following description: "
                f"{request.phrase}."
            )
        return self._run_inference(request, question, include_boxes=True)

    @modal.method()
    def point(self, request: PointRequest) -> InferenceResponse:
        question = f"Point to: {request.phrase}."
        return self._run_inference(
            request,
            question,
            include_boxes=False,
            include_points=True,
        )

    @modal.method()
    def detect_text(self, request: InferenceRequest) -> InferenceResponse:
        question = "Detect all the text in box format."
        return self._run_inference(request, question, include_boxes=True)

    @modal.method()
    def gui(self, request: GuiRequest) -> InferenceResponse:
        if request.output_type == "point":
            question = f"Point to: {request.phrase}."
            return self._run_inference(
                request,
                question,
                include_boxes=False,
                include_points=True,
            )
        question = (
            "Locate the region that matches the following description: "
            f"{request.phrase}."
        )
        return self._run_inference(request, question, include_boxes=True)

    @modal.method()
    def video_detect(self, request: VideoDetectRequest) -> VideoInferenceResponse:
        categories = "</c>".join(request.categories)
        question = (
            "Locate all the instances that matches the following description: "
            f"{categories}."
        )
        return self._run_video_inference(request, question, include_boxes=True)

    @modal.method()
    def video_ground(self, request: VideoGroundRequest) -> VideoInferenceResponse:
        if request.mode == "single":
            question = (
                "Locate a single instance that matches the following description: "
                f"{request.phrase}."
            )
        else:
            question = (
                "Locate all the instances that match the following description: "
                f"{request.phrase}."
            )
        return self._run_video_inference(request, question, include_boxes=True)

    @modal.method()
    def video_point(self, request: VideoPointRequest) -> VideoInferenceResponse:
        question = f"Point to: {request.phrase}."
        return self._run_video_inference(
            request,
            question,
            include_boxes=False,
            include_points=True,
        )

    @modal.method()
    def video_detect_text(
        self, request: VideoInferenceRequest
    ) -> VideoInferenceResponse:
        question = "Detect all the text in box format."
        return self._run_video_inference(request, question, include_boxes=True)

    @modal.method()
    def video_gui(self, request: VideoGuiRequest) -> VideoInferenceResponse:
        if request.output_type == "point":
            question = f"Point to: {request.phrase}."
            return self._run_video_inference(
                request,
                question,
                include_boxes=False,
                include_points=True,
            )
        question = (
            "Locate the region that matches the following description: "
            f"{request.phrase}."
        )
        return self._run_video_inference(request, question, include_boxes=True)

    @modal.asgi_app()
    def web(self):
        web_app = FastAPI(title="LocateAnything-3B API")
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @web_app.get("/health")
        async def health():
            return {"status": "ok", "model": MODEL_ID}

        @web_app.post("/detect", response_model=InferenceResponse)
        async def detect_endpoint(request: DetectRequest):
            return self.detect.local(request)

        @web_app.post("/ground", response_model=InferenceResponse)
        async def ground_endpoint(request: GroundRequest):
            return self.ground.local(request)

        @web_app.post("/point", response_model=InferenceResponse)
        async def point_endpoint(request: PointRequest):
            return self.point.local(request)

        @web_app.post("/detect_text", response_model=InferenceResponse)
        async def detect_text_endpoint(request: InferenceRequest):
            return self.detect_text.local(request)

        @web_app.post("/gui", response_model=InferenceResponse)
        async def gui_endpoint(request: GuiRequest):
            return self.gui.local(request)

        @web_app.post("/video/detect", response_model=VideoInferenceResponse)
        async def video_detect_endpoint(request: VideoDetectRequest):
            return self.video_detect.local(request)

        @web_app.post("/video/ground", response_model=VideoInferenceResponse)
        async def video_ground_endpoint(request: VideoGroundRequest):
            return self.video_ground.local(request)

        @web_app.post("/video/point", response_model=VideoInferenceResponse)
        async def video_point_endpoint(request: VideoPointRequest):
            return self.video_point.local(request)

        @web_app.post("/video/detect_text", response_model=VideoInferenceResponse)
        async def video_detect_text_endpoint(request: VideoInferenceRequest):
            return self.video_detect_text.local(request)

        @web_app.post("/video/gui", response_model=VideoInferenceResponse)
        async def video_gui_endpoint(request: VideoGuiRequest):
            return self.video_gui.local(request)

        return web_app


@app.local_entrypoint()
def main():
    print("LocateAnything-3B Modal app is ready.")
    print("Download weights: modal run modal/download.py::download_model")
    print("Dev server: modal serve modal/app.py")
    print("Deploy: modal deploy modal/app.py")
