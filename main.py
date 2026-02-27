import argparse
import ast
import csv
import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from src.fbvs import fbvs as fbvs_gaussian, fbvs_mesh, dvs_mesh, fbvs_redetect
from src.scene_ops import pose6_from_T


def get_frame_paths_and_pose(
    dataset_dir: str | Path, frame_idx: int
) -> tuple[Path, Path, np.ndarray]:
    """Return RGB path, depth path, and 6D pose for a dataset frame index."""
    data_dir = Path(dataset_dir)
    frame_name = f"frame-{int(frame_idx):06d}"

    rgb_path = data_dir / f"{frame_name}.color.jpg"
    depth_path = data_dir / f"{frame_name}.depth.png"
    pose_path = data_dir / f"{frame_name}.pose.txt"

    for p in (rgb_path, depth_path, pose_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing expected dataset file: {p}")

    T = np.loadtxt(pose_path, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(
            f"Pose file must contain a 4x4 matrix, got shape {T.shape} from {pose_path}"
        )

    pose = pose6_from_T(T, "c2w")
    return rgb_path, depth_path, pose


def append_fbvs_run_to_csv(
    csv_path: str | Path,
    *,
    desired_image_path: Path,
    desired_pose: np.ndarray,
    initial_pose: np.ndarray,
    converged: bool,
    iterations_before_converging: int,
    final_pose: np.ndarray,
    final_error: float,
) -> None:
    """Append one FBVS mesh run record to CSV (create with header if needed)."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_datetime[iso8601]",
        "desired_image_path[path]",
        "desired_pose[x,y,z(m),rx,ry,rz(rad)]",
        "initial_pose[x,y,z(m),rx,ry,rz(rad)]",
        "convergence_status[Success|Failed]",
        "iterations_before_converging[count]",
        "final_pose[x,y,z(m),rx,ry,rz(rad)]",
        "final_error[l2_norm]",
        "pose_diff_position[dx,dy,dz(m)]",
        "pose_diff_rotation[drx,dry,drz(rad)]",
        "position_euclidean_distance[m]",
        "geodesic_rotation_distance[rad]",
    ]

    desired_pose_arr = np.asarray(desired_pose, dtype=float).reshape(-1)
    final_pose_arr = np.asarray(final_pose, dtype=float).reshape(-1)
    pose_diff = desired_pose_arr - final_pose_arr
    pose_diff_position = pose_diff[:3]
    pose_diff_rotation = pose_diff[3:]
    position_euclidean_distance = float(np.linalg.norm(pose_diff_position))
    geodesic_rotation_distance = float("nan")
    if desired_pose_arr.size >= 6 and final_pose_arr.size >= 6:
        if np.all(np.isfinite(desired_pose_arr[3:6])) and np.all(np.isfinite(final_pose_arr[3:6])):
            geodesic_rotation_distance = geodesic_angle(
                desired_pose_arr[3:6], final_pose_arr[3:6]
            )

    row = {
        "run_datetime[iso8601]": datetime.now().isoformat(timespec="seconds"),
        "desired_image_path[path]": str(desired_image_path),
        "desired_pose[x,y,z(m),rx,ry,rz(rad)]": json.dumps(desired_pose_arr.tolist()),
        "initial_pose[x,y,z(m),rx,ry,rz(rad)]": json.dumps(
            np.asarray(initial_pose, dtype=float).tolist()
        ),
        "convergence_status[Success|Failed]": "Success" if bool(converged) else "Failed",
        "iterations_before_converging[count]": int(iterations_before_converging),
        "final_pose[x,y,z(m),rx,ry,rz(rad)]": json.dumps(final_pose_arr.tolist()),
        "final_error[l2_norm]": float(final_error),
        "pose_diff_position[dx,dy,dz(m)]": json.dumps(pose_diff_position.tolist()),
        "pose_diff_rotation[drx,dry,drz(rad)]": json.dumps(pose_diff_rotation.tolist()),
        "position_euclidean_distance[m]": position_euclidean_distance,
        "geodesic_rotation_distance[rad]": geodesic_rotation_distance,
    }

    def _canon(name: str) -> str:
        return name.strip().lower()

    aliases = {
        "run_datetime": "run_datetime[iso8601]",
        "run_datetime[iso8601]": "run_datetime[iso8601]",
        "desired_image_path": "desired_image_path[path]",
        "desired_image_path[path]": "desired_image_path[path]",
        "desired_pose": "desired_pose[x,y,z(m),rx,ry,rz(rad)]",
        "desired_pose[x,y,z(m),rx,ry,rz(rad)]": "desired_pose[x,y,z(m),rx,ry,rz(rad)]",
        "initial_pose": "initial_pose[x,y,z(m),rx,ry,rz(rad)]",
        "initial_pose[x,y,z(m),rx,ry,rz(rad)]": "initial_pose[x,y,z(m),rx,ry,rz(rad)]",
        "convergence_status": "convergence_status[Success|Failed]",
        "convergence_status[success|failed]": "convergence_status[Success|Failed]",
        "status": "convergence_status[Success|Failed]",
        "converged": "convergence_status[Success|Failed]",
        "iterations_before_converging": "iterations_before_converging[count]",
        "iterations_before_converging[count]": "iterations_before_converging[count]",
        "final_pose": "final_pose[x,y,z(m),rx,ry,rz(rad)]",
        "final_pose[x,y,z(m),rx,ry,rz(rad)]": "final_pose[x,y,z(m),rx,ry,rz(rad)]",
        "final_error": "final_error[l2_norm]",
        "final_error[l2_norm]": "final_error[l2_norm]",
        "pose_diff_position": "pose_diff_position[dx,dy,dz(m)]",
        "pose_diff_position[dx,dy,dz(m)]": "pose_diff_position[dx,dy,dz(m)]",
        "final_position_diff": "pose_diff_position[dx,dy,dz(m)]",
        "pose_diff_rotation": "pose_diff_rotation[drx,dry,drz(rad)]",
        "pose_diff_rotation[drx,dry,drz(rad)]": "pose_diff_rotation[drx,dry,drz(rad)]",
        "final_rotation_diff": "pose_diff_rotation[drx,dry,drz(rad)]",
        "position_euclidean_distance": "position_euclidean_distance[m]",
        "position_euclidean_distance[m]": "position_euclidean_distance[m]",
        "euclidean_distance": "position_euclidean_distance[m]",
        "euclidian_distance": "position_euclidean_distance[m]",
        "geodesic_rotation_distance": "geodesic_rotation_distance[rad]",
        "geodesic_rotation_distance[rad]": "geodesic_rotation_distance[rad]",
        "geodesic_angle": "geodesic_rotation_distance[rad]",
        "geodesic_angle[rad]": "geodesic_rotation_distance[rad]",
    }

    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_headers = reader.fieldnames or []
            existing_rows = list(reader)

        if existing_headers != fieldnames:
            migrated_rows = []
            for old_row in existing_rows:
                new_row = {k: "" for k in fieldnames}
                for old_key, old_value in old_row.items():
                    if old_key is None:
                        continue
                    mapped_key = aliases.get(_canon(old_key))
                    if mapped_key is None:
                        continue
                    if old_value is not None and old_value != "":
                        value = old_value
                        if mapped_key == "convergence_status[Success|Failed]":
                            v = str(old_value).strip().lower()
                            if v in {"true", "1", "yes", "success", "converged"}:
                                value = "Success"
                            elif v in {"false", "0", "no", "failed", "diverged"}:
                                value = "Failed"
                        new_row[mapped_key] = value
                migrated_rows.append(new_row)

            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(migrated_rows)

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_moge_model(
    *,
    device_name: str,
    model_version: str = "v2",
    pretrained_model_name_or_path: Optional[str] = None,
    use_fp16: bool = False,
) -> Any:
    """Load MoGe once so multiple FBVS runs can reuse the same model."""
    try:
        from moge.model import import_model_class_by_version
    except ImportError:
        local_moge = Path(__file__).resolve().parent / "MoGe"
        if local_moge.exists() and str(local_moge) not in sys.path:
            sys.path.insert(0, str(local_moge))
        from moge.model import import_model_class_by_version

    defaults = {
        "v1": "Ruicheng/moge-vitl",
        "v2": "Ruicheng/moge-2-vitl-normal",
    }
    if pretrained_model_name_or_path is None:
        if model_version not in defaults:
            raise ValueError(
                f"Unsupported model version '{model_version}'. Expected 'v1' or 'v2'."
            )
        pretrained_model_name_or_path = defaults[model_version]

    device = torch.device(device_name)
    model = (
        import_model_class_by_version(model_version)
        .from_pretrained(pretrained_model_name_or_path)
        .to(device)
        .eval()
    )
    if use_fp16 and hasattr(model, "half"):
        model.half()
    return model


def tweak_pose(pose: np.ndarray, tweak: float = 0.1) -> np.ndarray:
    """Perturb a 6D pose by element-wise percentage in [-tweak, +tweak]."""
    pose_arr = np.asarray(pose, dtype=np.float32).reshape(-1)
    if pose_arr.size != 6:
        raise ValueError(f"Input pose must have 6 elements, got shape {pose_arr.shape}.")
    scale = 1.0 + np.random.uniform(-float(tweak), float(tweak), size=6).astype(np.float32)
    return pose_arr * scale


def mesh_eval(
    scene_ply: str | Path,
    dataset_dir: str | Path,
    tests: int,
    runs_csv: str | Path,
    *,
    verbose: bool = False,
    pose_tweak: float = 0.1,
) -> None:
    """Run `fbvs_mesh` multiple times with random desired frame index in [0, 1100]."""
    dataset_dir = Path(dataset_dir)
    tests = int(tests)
    depth_device_name = "cuda" if torch.cuda.is_available() else "cpu"

    # Load MoGe once and reuse it across all trials.
    depth_model: Optional[Any] = None
    try:
        depth_model = load_moge_model(device_name=depth_device_name)
    except Exception as exc:
        print(f"[WARN] Failed to preload MoGe model once ({exc}). Falling back to per-run load.")

    for t in range(tests):
        sampled_frame_idx = int(np.random.randint(0, 1101))
        run_desired_rgb_path = dataset_dir / f"frame-{sampled_frame_idx:06d}.color.jpg"
        run_desired_pose = np.full(6, np.nan, dtype=np.float32)
        random_pose = np.full(6, np.nan, dtype=np.float32)

        try:
            run_desired_rgb_path, _, run_desired_pose = get_frame_paths_and_pose(
                dataset_dir, sampled_frame_idx
            )
            run_desired_pose = np.asarray(run_desired_pose, dtype=np.float32).reshape(-1)
            if verbose:
                print(
                    f"[mesh_eval] Test {t + 1}/{tests}: sampled desired frame idx={sampled_frame_idx}"
                )

            random_pose = tweak_pose(run_desired_pose, tweak=pose_tweak)
            run_metrics = fbvs_mesh(
                scene_ply=scene_ply,
                initial_pose=random_pose,
                desired_view=run_desired_rgb_path,
                error_tolerance=0.05,
                desired_pose=run_desired_pose,
                verbose=verbose,
                depth_model=depth_model,
                depth_device_name=depth_device_name,
                depth_resolution_level=6,
            )
            append_fbvs_run_to_csv(
                runs_csv,
                desired_image_path=run_desired_rgb_path,
                desired_pose=run_desired_pose,
                initial_pose=random_pose,
                converged=run_metrics["converged"],
                iterations_before_converging=run_metrics["iterations"],
                final_pose=run_metrics["final_pose"],
                final_error=run_metrics["final_error"],
            )
            print(
                f"[INFO] Test {t + 1}/{tests} completed: "
                f"{'Success' if run_metrics['converged'] else 'Failed'}"
            )
        except Exception as exc:
            append_fbvs_run_to_csv(
                runs_csv,
                desired_image_path=run_desired_rgb_path,
                desired_pose=run_desired_pose,
                initial_pose=random_pose,
                converged=False,
                iterations_before_converging=0,
                final_pose=np.full(6, np.nan, dtype=np.float32),
                final_error=float("nan"),
            )
            print(f"[WARN] Test {t + 1}/{tests} failed with exception: {exc}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _load_colmap_image_entries(images_txt_path: str | Path) -> list[dict[str, Any]]:
    """
    Parse COLMAP `images.txt` and return entries with image id, name, and c2w pose.
    """
    from scipy.spatial.transform import Rotation as R

    images_txt = Path(images_txt_path)
    if not images_txt.exists():
        raise FileNotFoundError(f"COLMAP images.txt not found: {images_txt}")

    entries: list[dict[str, Any]] = []
    with images_txt.open("r", encoding="utf-8") as f:
        while True:
            header = f.readline()
            if not header:
                break
            header = header.strip()
            if not header or header.startswith("#"):
                continue

            parts = header.split()
            if len(parts) < 10:
                raise ValueError(f"Malformed COLMAP image header line: '{header}'")

            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            image_name = parts[9]

            R_w2c = R.from_quat([qx, qy, qz, qw]).as_matrix()
            t_w2c = np.array([tx, ty, tz], dtype=np.float64)

            R_c2w = R_w2c.T
            C_world = -R_c2w @ t_w2c
            euler_c2w = R.from_matrix(R_c2w).as_euler("xyz")
            pose_c2w = np.array(
                [
                    C_world[0],
                    C_world[1],
                    C_world[2],
                    euler_c2w[0],
                    euler_c2w[1],
                    euler_c2w[2],
                ],
                dtype=np.float32,
            )

            entries.append(
                {
                    "image_id": image_id,
                    "image_name": image_name,
                    "pose_c2w": pose_c2w,
                }
            )

            # Skip POINTS2D line (the second line for this image entry).
            _ = f.readline()

    return entries


def gaussian_eval(
    scene_ply: str | Path,
    images_dir: str | Path,
    colmap_images_txt: str | Path,
    tests: int,
    runs_csv: str | Path,
    *,
    verbose: bool = False,
    pose_tweak: float = 0.1,
    K: Optional[np.ndarray] = None,
    error_tolerance: float = 0.035,
    max_iters: int = 20,
    use_redetect: bool = True,
) -> None:
    """
    Run Gaussian FBVS multiple times with random desired images from COLMAP `images.txt`.
    """
    scene_ply = Path(scene_ply)
    images_dir = Path(images_dir)
    tests = int(tests)

    entries = _load_colmap_image_entries(colmap_images_txt)
    if not entries:
        raise ValueError(f"No image entries parsed from {colmap_images_txt}")

    if verbose:
        print(
            f"[gaussian_eval] Loaded {len(entries)} COLMAP entries from {colmap_images_txt}"
        )
        print(
            f"[gaussian_eval] Controller={('fbvs_redetect' if use_redetect else 'fbvs')}, "
            f"tests={tests}, pose_tweak={pose_tweak}"
        )

    controller = fbvs_redetect if use_redetect else fbvs_gaussian

    for t in range(tests):
        sampled = entries[int(np.random.randint(0, len(entries)))]
        run_desired_rgb_path = images_dir / str(sampled["image_name"])
        run_desired_pose = np.asarray(sampled["pose_c2w"], dtype=np.float32).reshape(-1)
        random_pose = np.full(6, np.nan, dtype=np.float32)

        try:
            if not run_desired_rgb_path.exists():
                raise FileNotFoundError(f"Missing desired image file: {run_desired_rgb_path}")

            if verbose:
                print(
                    f"[gaussian_eval] Test {t + 1}/{tests}: "
                    f"image_id={sampled['image_id']}, image={sampled['image_name']}"
                )

            random_pose = tweak_pose(run_desired_pose, tweak=pose_tweak)
            run_metrics = controller(
                scene=scene_ply,
                initial_pose=random_pose,
                desired_view=run_desired_rgb_path,
                error_tolerance=error_tolerance,
                desired_pose=run_desired_pose,
                verbose=verbose,
                K=K,
                max_iters=max_iters,
                save_frames=False,
                save_video=False,
            )
            append_fbvs_run_to_csv(
                runs_csv,
                desired_image_path=run_desired_rgb_path,
                desired_pose=run_desired_pose,
                initial_pose=random_pose,
                converged=run_metrics["converged"],
                iterations_before_converging=run_metrics["iterations"],
                final_pose=run_metrics["final_pose"],
                final_error=run_metrics["final_error"],
            )
            print(
                f"[INFO] Gaussian test {t + 1}/{tests} completed: "
                f"{'Success' if run_metrics['converged'] else 'Failed'}"
            )
        except Exception as exc:
            append_fbvs_run_to_csv(
                runs_csv,
                desired_image_path=run_desired_rgb_path,
                desired_pose=run_desired_pose,
                initial_pose=random_pose,
                converged=False,
                iterations_before_converging=0,
                final_pose=np.full(6, np.nan, dtype=np.float32),
                final_error=float("nan"),
            )
            print(f"[WARN] Gaussian test {t + 1}/{tests} failed with exception: {exc}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _row_get(row: dict[str, str], *keys: str) -> str:
    for k in keys:
        if k in row and row[k] is not None:
            return str(row[k]).strip()
    return ""


def _parse_float_cell(value: str) -> float:
    if value is None:
        return float("nan")
    s = str(value).strip()
    if s == "":
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def _parse_vector_cell(value: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            arr = np.asarray(parser(s), dtype=float).reshape(-1)
            if arr.size > 0:
                return arr
        except Exception:
            pass

    s_core = s.strip("[]")
    for sep in (",", " "):
        arr = np.fromstring(s_core, sep=sep, dtype=float)
        if arr.size > 0:
            return arr
    return None


def _stat_block(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _corr(x: list[float], y: list[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if np.sum(mask) < 2:
        return float("nan")
    if np.std(x_arr[mask]) < 1e-12 or np.std(y_arr[mask]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_arr[mask], y_arr[mask])[0, 1])


def _fmt3(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if np.isnan(v):
            return "nan"
        if np.isinf(v):
            return "inf" if v > 0 else "-inf"
        if percent:
            return f"{100.0 * v:.3f}%"
        return f"{v:.3f}"
    return str(value)


def _render_table(headers: list[str], rows: list[list[Any]]) -> str:
    table_rows = [[_fmt3(v) for v in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line(parts: list[str]) -> str:
        return "| " + " | ".join(parts[i].ljust(widths[i]) for i in range(len(parts))) + " |"

    header_line = _line(headers)
    sep_line = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    body_lines = [_line(row) for row in table_rows]
    return "\n".join([header_line, sep_line, *body_lines])


def summarize_mesh_eval(csv_path: str | Path, *, print_summary: bool = True) -> dict[str, Any]:
    """Summarize FBVS mesh robustness metrics from the run CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    records: list[dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        status_str = _row_get(
            row,
            "convergence_status[Success|Failed]",
            "convergence_status",
            "status",
            "converged",
        ).lower()

        success: Optional[bool]
        if status_str in {"success", "true", "1", "yes", "converged"}:
            success = True
        elif status_str in {"failed", "false", "0", "no", "diverged"}:
            success = False
        else:
            success = None

        desired_pose = _parse_vector_cell(
            _row_get(
                row,
                "desired_pose[x,y,z(m),rx,ry,rz(rad)]",
                "desired_pose",
            )
        )
        initial_pose = _parse_vector_cell(
            _row_get(
                row,
                "initial_pose[x,y,z(m),rx,ry,rz(rad)]",
                "initial_pose",
            )
        )
        final_pose = _parse_vector_cell(
            _row_get(
                row,
                "final_pose[x,y,z(m),rx,ry,rz(rad)]",
                "final_pose",
            )
        )
        pose_diff_pos = _parse_vector_cell(
            _row_get(
                row,
                "pose_diff_position[dx,dy,dz(m)]",
                "pose_diff_position",
                "final_position_diff",
            )
        )
        pose_diff_rot = _parse_vector_cell(
            _row_get(
                row,
                "pose_diff_rotation[drx,dry,drz(rad)]",
                "pose_diff_rotation",
                "final_rotation_diff",
            )
        )

        if pose_diff_pos is None and desired_pose is not None and final_pose is not None:
            if desired_pose.size >= 3 and final_pose.size >= 3:
                pose_diff_pos = desired_pose[:3] - final_pose[:3]
        if pose_diff_rot is None and desired_pose is not None and final_pose is not None:
            if desired_pose.size >= 6 and final_pose.size >= 6:
                pose_diff_rot = desired_pose[3:6] - final_pose[3:6]

        final_error = _parse_float_cell(
            _row_get(
                row,
                "final_error[l2_norm]",
                "final_error",
            )
        )
        iterations = _parse_float_cell(
            _row_get(
                row,
                "iterations_before_converging[count]",
                "iterations_before_converging",
            )
        )
        pos_dist = _parse_float_cell(
            _row_get(
                row,
                "position_euclidean_distance[m]",
                "position_euclidean_distance",
                "euclidean_distance",
                "euclidian_distance",
            )
        )
        if not np.isfinite(pos_dist) and pose_diff_pos is not None and pose_diff_pos.size >= 3:
            pos_dist = float(np.linalg.norm(pose_diff_pos[:3]))

        rot_norm = float("nan")
        if pose_diff_rot is not None and pose_diff_rot.size >= 3:
            rot_norm = float(np.linalg.norm(pose_diff_rot[:3]))

        geodesic_rot_dist = _parse_float_cell(
            _row_get(
                row,
                "geodesic_rotation_distance[rad]",
                "geodesic_rotation_distance",
                "geodesic_angle[rad]",
                "geodesic_angle",
            )
        )
        if (
            not np.isfinite(geodesic_rot_dist)
            and desired_pose is not None
            and final_pose is not None
            and desired_pose.size >= 6
            and final_pose.size >= 6
            and np.all(np.isfinite(desired_pose[3:6]))
            and np.all(np.isfinite(final_pose[3:6]))
        ):
            geodesic_rot_dist = geodesic_angle(desired_pose[3:6], final_pose[3:6])

        init_pos_perturb = float("nan")
        init_rot_perturb = float("nan")
        if desired_pose is not None and initial_pose is not None:
            if desired_pose.size >= 3 and initial_pose.size >= 3:
                init_pos_perturb = float(np.linalg.norm(initial_pose[:3] - desired_pose[:3]))
            if desired_pose.size >= 6 and initial_pose.size >= 6:
                init_rot_perturb = float(np.linalg.norm(initial_pose[3:6] - desired_pose[3:6]))

        records.append(
            {
                "row": i,
                "run_datetime": _row_get(row, "run_datetime[iso8601]", "run_datetime"),
                "success": success,
                "iterations": iterations,
                "final_error": final_error,
                "pos_dist": pos_dist,
                "rot_norm": rot_norm,
                "geodesic_rot_dist": geodesic_rot_dist,
                "pose_diff_pos": pose_diff_pos,
                "pose_diff_rot": pose_diff_rot,
                "init_pos_perturb": init_pos_perturb,
                "init_rot_perturb": init_rot_perturb,
            }
        )

    total = len(records)
    success_records = [r for r in records if r["success"] is True]
    failed_records = [r for r in records if r["success"] is False]

    success_count = len(success_records)
    failed_count = len(failed_records)
    success_rate = (success_count / total) if total > 0 else float("nan")

    iter_success = [r["iterations"] for r in success_records]
    err_success = [r["final_error"] for r in success_records]
    pos_success = [r["pos_dist"] for r in success_records]
    rot_success = [r["rot_norm"] for r in success_records]
    geodesic_success = [r["geodesic_rot_dist"] for r in success_records]
    init_pos_success = [r["init_pos_perturb"] for r in success_records]
    init_rot_success = [r["init_rot_perturb"] for r in success_records]

    pos_bias = np.array(
        [
            r["pose_diff_pos"][:3]
            for r in success_records
            if r["pose_diff_pos"] is not None and r["pose_diff_pos"].size >= 3
        ],
        dtype=float,
    )
    rot_bias = np.array(
        [
            r["pose_diff_rot"][:3]
            for r in success_records
            if r["pose_diff_rot"] is not None and r["pose_diff_rot"].size >= 3
        ],
        dtype=float,
    )

    axis_bias: dict[str, Any] = {}
    if pos_bias.size > 0:
        axis_bias["position_mean_m"] = {
            "dx": float(np.mean(pos_bias[:, 0])),
            "dy": float(np.mean(pos_bias[:, 1])),
            "dz": float(np.mean(pos_bias[:, 2])),
        }
        axis_bias["position_std_m"] = {
            "dx": float(np.std(pos_bias[:, 0])),
            "dy": float(np.std(pos_bias[:, 1])),
            "dz": float(np.std(pos_bias[:, 2])),
        }
    if rot_bias.size > 0:
        axis_bias["rotation_mean_rad"] = {
            "drx": float(np.mean(rot_bias[:, 0])),
            "dry": float(np.mean(rot_bias[:, 1])),
            "drz": float(np.mean(rot_bias[:, 2])),
        }
        axis_bias["rotation_std_rad"] = {
            "drx": float(np.std(rot_bias[:, 0])),
            "dry": float(np.std(rot_bias[:, 1])),
            "drz": float(np.std(rot_bias[:, 2])),
        }

    worst_position = None
    worst_rotation = None
    finite_pos_records = [r for r in success_records if np.isfinite(r["pos_dist"])]
    finite_rot_records = [r for r in success_records if np.isfinite(r["rot_norm"])]
    if finite_pos_records:
        wr = max(finite_pos_records, key=lambda r: r["pos_dist"])
        worst_position = {
            "row": wr["row"],
            "run_datetime": wr["run_datetime"],
            "position_euclidean_distance_m": float(wr["pos_dist"]),
            "final_error": float(wr["final_error"]) if np.isfinite(wr["final_error"]) else float("nan"),
        }
    if finite_rot_records:
        wr = max(finite_rot_records, key=lambda r: r["rot_norm"])
        worst_rotation = {
            "row": wr["row"],
            "run_datetime": wr["run_datetime"],
            "rotation_residual_norm_rad": float(wr["rot_norm"]),
            "final_error": float(wr["final_error"]) if np.isfinite(wr["final_error"]) else float("nan"),
        }

    summary = {
        "csv_path": str(csv_path),
        "runs": {
            "total": total,
            "success_count": success_count,
            "failed_count": failed_count,
            "success_rate": float(success_rate),
        },
        "iterations_success": _stat_block(iter_success),
        "final_error_success": _stat_block(err_success),
        "position_distance_success_m": _stat_block(pos_success),
        "rotation_residual_success_rad": _stat_block(rot_success),
        "geodesic_rotation_distance_success_rad": _stat_block(geodesic_success),
        "initial_position_perturbation_success_m": _stat_block(init_pos_success),
        "initial_rotation_perturbation_success_rad": _stat_block(init_rot_success),
        "correlations_success": {
            "init_position_perturb_vs_final_error": _corr(init_pos_success, err_success),
            "init_position_perturb_vs_iterations": _corr(init_pos_success, iter_success),
            "init_rotation_perturb_vs_final_error": _corr(init_rot_success, err_success),
            "init_rotation_perturb_vs_iterations": _corr(init_rot_success, iter_success),
        },
        "axis_bias": axis_bias,
        "worst_case_success": {
            "position": worst_position,
            "rotation": worst_rotation,
        },
    }

    if print_summary:
        pos_block = summary["position_distance_success_m"]
        err_block = summary["final_error_success"]
        geodesic_block = summary["geodesic_rotation_distance_success_rad"]

        print("\n=== Mesh Eval Summary ===")
        print(f"nb_data_available: {total}")
        print(f"success: {success_count}")
        print(f"failure: {failed_count}")
        print(
            f"euclidean_distance_mean_std: "
            f"{_fmt3(pos_block.get('mean', float('nan')))} +/- {_fmt3(pos_block.get('std', float('nan')))}"
        )
        print(
            f"final_error_mean_std: "
            f"{_fmt3(err_block.get('mean', float('nan')))} +/- {_fmt3(err_block.get('std', float('nan')))}"
        )
        print(
            f"geodesic_rotation_distance_mean_std: "
            f"{_fmt3(geodesic_block.get('mean', float('nan')))} +/- {_fmt3(geodesic_block.get('std', float('nan')))}"
        )

    return summary


def geodesic_angle(r1, r2) -> float:
    from scipy.spatial.transform import Rotation as R
    R1 = R.from_euler("xyz", r1).as_matrix()
    R2 = R.from_euler("xyz", r2).as_matrix()
    Rrel = R1.T @ R2
    c = (np.trace(Rrel) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))  # radians in [0, pi]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show live current-vs-target view while FBVS mesh simulation runs.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/apt1/living/data"),
        help="Directory containing frame-XXXXXX.{color.jpg,depth.png,pose.txt}",
    )
    parser.add_argument(
        "--scene-ply",
        type=Path,
        default=Path("data/apt1/living/aliving.ply"),
        help="Path to mesh (.ply) used by fbvs_mesh.",
    )
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=Path("logs/fbvs_mesh_runs_05.csv"),
        help="CSV file where each fbvs_mesh run is appended.",
    )
    parser.add_argument(
        "--tests",
        type=int,
        default=1,
        help="Number of randomized FBVS evaluations to run.",
    )
    parser.add_argument(
        "--pose-tweak",
        type=float,
        default=0.1,
        help="Relative random perturbation magnitude for each pose element.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print robustness summary from the runs CSV after evaluation.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary from --runs-csv and exit.",
    )
    args = parser.parse_args()

    if args.summary_only:
        summarize_mesh_eval(args.runs_csv, print_summary=True)
        raise SystemExit(0)

    if args.verbose:
        print(f"[main] dataset dir: {args.dataset_dir}")
        print("[main] desired frame idx is sampled randomly in [0, 1100] per trial.")

    # Example batch evaluation:
    # python main.py --tests 20 --pose-tweak 0.15 --runs-csv logs/fbvs_mesh_runs_05.csv
    # mesh_eval(
    #     scene_ply=args.scene_ply,
    #     dataset_dir=args.dataset_dir,
    #     tests=args.tests,
    #     runs_csv=args.runs_csv,
    #     verbose=args.verbose,
    #     pose_tweak=args.pose_tweak,
    # )
    #
    ###################################################################################################################
    # Example Gaussian evaluation:
    ###################################################################################################################

    # # python main.py --tests 20 --pose-tweak 0.10 --runs-csv logs/fbvs_gaussian_runs.csv
    gaussian_eval(
        scene_ply=Path("data/playroom/playroom.ply"),
        images_dir=Path("data/playroom/images"),
        colmap_images_txt=Path("data/playroom/info/images.txt"),
        tests=args.tests,
        runs_csv=Path("logs/fbvs_gaussian_runs.csv"),
        verbose=args.verbose,
        pose_tweak=args.pose_tweak,
        K=np.array(
            [
                [1040.0073037593279, 0.0, 632.0],
                [0.0, 1040.1927566661841, 416.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        use_redetect=True,
    )
    # if args.summary:
    #     summarize_mesh_eval(args.runs_csv, print_summary=True)
    #
    # # Example single FBVS simulation (from a given initial pose to a target image) with verbose ON.
    # # Set this to True when you want to run exactly one simulation example.
    # """
    # python main.py --tests 50 --pose-tweak 0.10 --dataset-dir data/apt1/kitchen/data --runs-csv logs/fbvs_mesh_runs_10.csv
    # """
    # example_scene_ply = Path("data/apt1/living/aliving.ply")
    # example_dataset_dir = Path("data/apt1/living/data")
    # example_frame_idx = 13
    #
    # desired_rgb_path, _, desired_pose = get_frame_paths_and_pose(
    #     example_dataset_dir, example_frame_idx
    # )
    # initial_pose = np.array(
    #     [0, 0.1, 1.3, -2.20, -0.2,
    #      -0.7], dtype=np.float32
    # )
    #
    # run_metrics = fbvs_mesh(
    #     scene_ply=example_scene_ply,
    #     initial_pose=initial_pose,
    #     desired_view=desired_rgb_path,
    #     error_tolerance=0.025,
    #     desired_pose=desired_pose,
    #     verbose=True,
    #     save_frames=True,
    #     frames_dir="debug_frames/"
    # )
    # print(f"[single_fbvs_example] {run_metrics}")
    # final_pose = np.asarray(run_metrics["final_pose"], dtype=np.float32).reshape(-1)
    # desired_pose_arr = np.asarray(desired_pose, dtype=np.float32).reshape(-1)
    # pos_distance_m = float(np.linalg.norm(desired_pose_arr[:3] - final_pose[:3]))
    # geodesic_rot_rad = geodesic_angle(desired_pose_arr[3:6], final_pose[3:6])
    # print(
    #     f"[single_fbvs_example] position_distance_m={pos_distance_m:.6f}, "
    #     f"geodesic_rotation_rad={geodesic_rot_rad:.6f}"
    # )
    # dvs_mesh(
    #     scene_ply=example_scene_ply,
    #     initial_pose=desired_pose,
    #     desired_view=desired_rgb_path,
    #     error_tolerance=0.025,
    #     desired_pose=desired_pose,
    #     verbose=True,
    #     save_frames=True,
    #     frames_dir="debug_frames/"
    # )
    ###################################################################################################################
    # FBVS Gaussian Playroom Example
    ###################################################################################################################
    # from src.servoing.gaussian_fbvs import fbvs as fbvs_gaussian, get_pose_colmap
    #
    # gaussian_scene_ply = Path("data/playroom/playroom.ply")
    # gaussian_desired_view = Path("data/playroom/images/DSC05574.jpg")
    # gaussian_colmap_images = Path("data/playroom/info/images.txt")
    # gaussian_target_image_id = 3
    # #
    # K_gaussian = np.array(
    #     [
    #         [1040.0073037593279, 0.0, 632.0],
    #         [0.0, 1040.1927566661841, 416.0],
    #         [0.0, 0.0, 1.0],
    #     ],
    #     dtype=np.float32,
    # )
    # #
    # desired_pose_gaussian = get_pose_colmap(
    #     gaussian_target_image_id, path=gaussian_colmap_images
    # )
    # if desired_pose_gaussian is None:
    #     raise ValueError(
    #         f"Could not find image_id={gaussian_target_image_id} in {gaussian_colmap_images}"
    #     )
    #
    # # Start from a perturbed pose so FBVS has a non-zero control objective.
    # initial_pose_gaussian = tweak_pose(desired_pose_gaussian, tweak=0.10)
    #
    # run_metrics_gaussian = fbvs_gaussian(
    #     scene=gaussian_scene_ply,
    #     initial_pose=np.array([ 3.226352  ,  2.006508  , -2.9522364 , -0.23189576,  0.01949945,
    #     0.42474204]),
    #     desired_view=gaussian_desired_view,
    #     K=K_gaussian,
    #     desired_pose=desired_pose_gaussian,
    #     error_tolerance=0.035,
    #     max_iters=20,
    #     verbose=True,
    #     save_frames=True,
    #     frames_dir="debug_frames",
    #     save_video=True,
    # )
    # print(f"[single_fbvs_gaussian_example] {run_metrics_gaussian}")
    ###################################################################################################################

    ###################################################################################################################
    # FBVS Gaussian Playroom Example (Re-detect + Re-match each iteration)
    ###################################################################################################################
    # run_metrics_gaussian_redetect = fbvs_redetect(
    #     scene=gaussian_scene_ply,
    #     initial_pose=np.array([ 3.226352  ,  2.006508  , -2.9522364 , -0.23189576,  0.91949945,
    #     0.42474204]),
    #     desired_view=gaussian_desired_view,
    #     K=K_gaussian,
    #     desired_pose=desired_pose_gaussian,
    #     error_tolerance=0.035,
    #     max_iters=50,
    #     verbose=True,
    #     save_frames=True,
    #     frames_dir="debug_frames",
    #     save_video=True,
    # )
    # print(f"[single_fbvs_gaussian_redetect_example] {run_metrics_gaussian_redetect}")
