"""SAM3 instance masks joined to pixel-aligned Isaac camera point maps."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class InstanceSegmentation:
    """Original-resolution instance predictions from SAM3."""

    masks: np.ndarray
    boxes_xyxy: np.ndarray
    scores: np.ndarray

    def validate(self, image_shape: tuple[int, int]) -> None:
        masks = np.asarray(self.masks)
        boxes = np.asarray(self.boxes_xyxy)
        scores = np.asarray(self.scores)
        if masks.ndim != 3 or masks.shape[1:] != image_shape:
            raise ValueError(f"masks must have shape (N, {image_shape[0]}, {image_shape[1]}), got {masks.shape}")
        if boxes.shape != (masks.shape[0], 4):
            raise ValueError(f"boxes must have shape ({masks.shape[0]}, 4), got {boxes.shape}")
        if scores.shape != (masks.shape[0],):
            raise ValueError(f"scores must have shape ({masks.shape[0]},), got {scores.shape}")


def infer_sam3_instances(
    rgb: np.ndarray,
    prompt: str,
    *,
    model_id: str = "facebook/sam3",
    device: str = "cuda",
    dtype: str = "fp16",
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    local_files_only: bool = True,
) -> InstanceSegmentation:
    """Run the Hugging Face SAM3 text-only inference recipe."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if dtype not in {"fp16", "fp32"}:
        raise ValueError(f"unsupported dtype: {dtype}")
    if device == "cpu" and dtype == "fp16":
        raise ValueError("CPU inference requires --sam3-dtype fp32")

    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    processor = Sam3Processor.from_pretrained(model_id, local_files_only=local_files_only)
    model = Sam3Model.from_pretrained(model_id, local_files_only=local_files_only)
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    model = model.to(device=device, dtype=torch_dtype).eval()

    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=score_threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks = result["masks"].detach().to("cpu").numpy().astype(bool, copy=False)
    boxes = result["boxes"].detach().to("cpu").numpy().astype(np.float32, copy=False)
    scores = result["scores"].detach().to("cpu").numpy().astype(np.float32, copy=False)
    prediction = InstanceSegmentation(masks=masks, boxes_xyxy=boxes, scores=scores)
    prediction.validate(tuple(rgb.shape[:2]))
    return prediction


def select_masked_points(
    mask: np.ndarray,
    depth_m: np.ndarray,
    points_camera: np.ndarray,
    points_world: np.ndarray,
    rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select valid pixel-aligned points without reimplementing projection."""
    mask = np.asarray(mask, dtype=bool)
    depth_m = np.asarray(depth_m)
    points_camera = np.asarray(points_camera)
    points_world = np.asarray(points_world)
    rgb = np.asarray(rgb)
    if mask.shape != depth_m.shape:
        raise ValueError(f"mask shape {mask.shape} does not match depth {depth_m.shape}")
    expected_points_shape = (*depth_m.shape, 3)
    if points_camera.shape != expected_points_shape or points_world.shape != expected_points_shape:
        raise ValueError("point maps must be pixel-aligned HxWx3 arrays")
    if rgb.shape != expected_points_shape:
        raise ValueError(f"rgb must have shape {expected_points_shape}, got {rgb.shape}")

    valid = (
        mask
        & np.isfinite(depth_m)
        & (depth_m > 0.0)
        & np.all(np.isfinite(points_camera), axis=2)
        & np.all(np.isfinite(points_world), axis=2)
    )
    return valid, points_camera[valid], points_world[valid], rgb[valid]


def point_statistics(points: np.ndarray) -> dict[str, list[float] | int | None]:
    """Return inspectable metric bounds for a point cloud."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if points.shape[0] == 0:
        return {"count": 0, "centroid_m": None, "min_m": None, "max_m": None}
    return {
        "count": int(points.shape[0]),
        "centroid_m": np.mean(points, axis=0).astype(float).tolist(),
        "min_m": np.min(points, axis=0).astype(float).tolist(),
        "max_m": np.max(points, axis=0).astype(float).tolist(),
    }


def write_ascii_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a dependency-free, standard PLY point cloud for manual inspection."""
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("points and colors must both have shape (N, 3)")
    path = Path(path)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {points.shape[0]}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("end_header\n")
        for point, color in zip(points, colors):
            stream.write(
                f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def save_segmentation_artifacts(
    output: str | Path,
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    points_camera: np.ndarray,
    points_world: np.ndarray,
    prediction: InstanceSegmentation,
    prompt: str,
    model_id: str,
    score_threshold: float,
    mask_threshold: float,
) -> dict:
    """Save masks, overlays and unmodified masked point clouds."""
    from PIL import Image

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prediction.validate(tuple(rgb.shape[:2]))
    if prediction.masks.shape[0] == 0:
        union_mask = np.zeros(depth_m.shape, dtype=bool)
    else:
        union_mask = np.any(prediction.masks, axis=0)
    valid_mask, camera_points, world_points, colors = select_masked_points(
        union_mask, depth_m, points_camera, points_world, rgb
    )

    np.save(output / "instance_masks.npy", prediction.masks)
    np.save(output / "union_mask.npy", union_mask)
    np.save(output / "valid_3d_mask.npy", valid_mask)
    np.save(output / "points_camera.npy", camera_points.astype(np.float32, copy=False))
    np.save(output / "points_world.npy", world_points.astype(np.float32, copy=False))
    np.save(output / "colors.npy", colors.astype(np.uint8, copy=False))
    Image.fromarray((union_mask.astype(np.uint8) * 255), mode="L").save(output / "mask.png")
    overlay = np.asarray(rgb, dtype=np.float32).copy()
    overlay[union_mask] = overlay[union_mask] * 0.45 + np.array([0.0, 255.0, 80.0]) * 0.55
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(output / "overlay.png")
    write_ascii_ply(output / "points_camera.ply", camera_points, colors)
    write_ascii_ply(output / "points_world.ply", world_points, colors)

    instances = [
        {
            "index": index,
            "score": float(prediction.scores[index]),
            "box_xyxy_px": prediction.boxes_xyxy[index].astype(float).tolist(),
            "mask_pixels": int(np.count_nonzero(prediction.masks[index])),
        }
        for index in range(prediction.masks.shape[0])
    ]
    report = {
        "reference": "Hugging Face Transformers SAM3 text-only instance segmentation",
        "model_id": model_id,
        "prompt": prompt,
        "score_threshold": score_threshold,
        "mask_threshold": mask_threshold,
        "instance_count": len(instances),
        "instances": instances,
        "union_mask_pixels": int(np.count_nonzero(union_mask)),
        "valid_3d_pixels": int(np.count_nonzero(valid_mask)),
        "camera_points": point_statistics(camera_points),
        "world_points": point_statistics(world_points),
        "automatic_checks_passed": bool(len(instances) > 0 and camera_points.shape[0] > 0),
        "manual_review_required": True,
    }
    (output / "segmentation_check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
