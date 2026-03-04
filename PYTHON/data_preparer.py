# data preparer
import os
import csv
import json
from pathlib import Path

import cv2
import numpy as np


POSITION_COLUMN_CANDIDATES = (
    "pipette_z_microns",
    "defocus_microns",
    "pipette_z",
    "z_microns",
    "z",
)
DEFAULT_GABOR_THETAS = (
    0.0,
    np.pi / 4.0,
    np.pi / 2.0,
    3.0 * np.pi / 4.0,
)


def _validate_odd_kernel(name: str, value: int) -> None:
    if value < 1 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer.")

def zero_initial_position(movement_data):
    """
    Adjust movement data so that the initial stage and pipette positions are zeroed.
    """
    if not movement_data:
        return movement_data
    initial_stage = movement_data[0]['stage']
    initial_pipette = movement_data[0]['pipette']
    for record in movement_data:
        record['stage'] = tuple(s - i for s, i in zip(record['stage'], initial_stage))
        record['pipette'] = tuple(p - i for p, i in zip(record['pipette'], initial_pipette))
    return movement_data


def load_movement_data(file_path):
    """
    Load movement data from the given file.
    
    Supports two formats:
      1. New format: CSV file with semicolon delimiter and a header containing "timestamp".
         Expected columns: timestamp, st_x, st_y, st_z, pi_x, pi_y, pi_z.
      2. Old format: Space-delimited file with colon-separated key-value pairs.
         Expected line format:
           time:123.456 stage_x:... stage_y:... stage_z:... pipette_x:... pipette_y:... pipette_z:...
    
    No scaling is applied.
    """
    movement_data = []
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        # Detect new format: CSV with semicolon delimiter and header containing "timestamp"
        if ';' in first_line and 'timestamp' in first_line.lower():
            file.seek(0)
            reader = csv.DictReader(file, delimiter=';')
            for row in reader:
                try:
                    time_value = float(row['timestamp'])
                    st_x = float(row['st_x'])
                    st_y = float(row['st_y'])
                    st_z = float(row['st_z'])
                    pi_x = float(row['pi_x'])
                    pi_y = float(row['pi_y'])
                    pi_z = float(row['pi_z'])
                    movement_data.append({
                        'time': time_value,
                        'stage': (st_x, st_y, st_z),
                        'pipette': (pi_x, pi_y, pi_z)
                    })
                except Exception as e:
                    print(f"Skipping invalid row: {row}. Error: {e}")
        else:
            # Assume old format with space-delimited values and colon-split key-value pairs
            file.seek(0)
            for line in file:
                parts = [part for part in line.strip().split(' ') if part]
                if len(parts) < 7:
                    print(f"Skipping invalid data line: {line}")
                    continue
                try:
                    time_value = float(parts[0].split(':')[1])
                    st_x = float(parts[1].split(':')[1])
                    st_y = float(parts[2].split(':')[1])
                    st_z = float(parts[3].split(':')[1])
                    pi_x = float(parts[4].split(':')[1])
                    pi_y = float(parts[5].split(':')[1])
                    pi_z = float(parts[6].split(':')[1])
                    movement_data.append({
                        'time': time_value,
                        'stage': (st_x, st_y, st_z),
                        'pipette': (pi_x, pi_y, pi_z)
                    })
                except Exception as e:
                    print(f"Skipping invalid data line: {line}. Error: {e}")
    # Sort by time to simplify further processing
    movement_data.sort(key=lambda r: r['time'])
    return zero_initial_position(movement_data)

def extract_image_data(image_filename):
    """
    Extract image index and timestamp from the filename.
    Expected filename format: <index>_<timestamp>.<ext>
    e.g., "1_123.456.png"
    """
    try:
        base = os.path.basename(image_filename)
        parts = base.split('_')
        if len(parts) < 2:
            raise ValueError("Filename does not contain the expected underscore format.")
        # The first part is an index (unused here) and the second is the timestamp.
        index = int(parts[0])
        timestamp_str = parts[1].rsplit('.', 1)[0]
        timestamp = float(timestamp_str)
        return index, timestamp
    except Exception as e:
        print(f"Error extracting data from {image_filename}: {e}")
        return 0, 0.0

def find_closest_movement_record(image_timestamp, movement_data):
    """
    Return the movement record whose timestamp is closest to the given image_timestamp.
    """
    return min(movement_data, key=lambda record: abs(record['time'] - image_timestamp))


def load_calibration_matrix_xy(calibration_path):
    """
    Load the manip/pipette affine matrix and return its XY projection (2x3).
    """
    if not os.path.exists(calibration_path):
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    with open(calibration_path, "r", encoding="utf-8") as calibration_file:
        calibration = json.load(calibration_file)

    manip_entry = calibration.get("manip") or calibration.get("pipette") or calibration
    if "M" not in manip_entry:
        raise ValueError("Calibration file missing manip/pipette matrix 'M'.")

    matrix = np.asarray(manip_entry["M"], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 3:
        raise ValueError(f"Unexpected calibration matrix shape: {matrix.shape}")
    return matrix[:2, :3]


def crop_tip_roi_256(
    image_path: str,
    output_path: str,
    manip_matrix_xy: np.ndarray,
    zeroed_pipette_xyz: np.ndarray,
    crop_size: int = 256,
) -> tuple[bool, str]:
    """
    Crop a fixed square ROI around projected pipette tip position.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return False, "unreadable_image"

    image_height, image_width = image.shape[:2]
    if image_width < crop_size or image_height < crop_size:
        return False, "image_too_small"

    pixel_delta_xy = manip_matrix_xy @ zeroed_pipette_xyz
    center_x = (image_width / 2.0) + float(pixel_delta_xy[0])
    center_y = (image_height / 2.0) + float(pixel_delta_xy[1])

    half = crop_size / 2.0
    x1 = int(round(center_x - half))
    y1 = int(round(center_y - half))

    # Clamp to image bounds so we keep a fixed crop size whenever possible.
    x1 = max(0, min(x1, image_width - crop_size))
    y1 = max(0, min(y1, image_height - crop_size))
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    crop = image[y1:y2, x1:x2]
    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        return False, "invalid_crop_shape"

    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    if not cv2.imwrite(output_path, crop):
        return False, "write_failed"

    return True, "ok"


def detect_focus_position_column(csv_path: str | Path, requested_column: str | None = None) -> str:
    """
    Detect which CSV column should be treated as the recorded position for plotting.
    """
    csv_path = Path(csv_path)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

    if not fieldnames:
        raise ValueError(f"CSV has no header: {csv_path}")
    if "filename" not in fieldnames:
        raise ValueError(f"CSV missing required 'filename' column: {csv_path}")

    if requested_column:
        if requested_column not in fieldnames:
            raise ValueError(
                f"Requested position column '{requested_column}' not found. "
                f"Available columns: {fieldnames}"
            )
        return requested_column

    for candidate in POSITION_COLUMN_CANDIDATES:
        if candidate in fieldnames:
            return candidate

    raise ValueError(
        "Could not detect position column. "
        f"Checked: {POSITION_COLUMN_CANDIDATES}. Available: {fieldnames}"
    )


def resolve_focus_images_dir(csv_path: str | Path, images_dir_arg: str | None = None) -> Path:
    """
    Resolve the folder containing images listed in a final pipette_z_data CSV.
    """
    if images_dir_arg:
        images_dir = Path(images_dir_arg).expanduser().resolve()
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {images_dir}")
        return images_dir

    csv_path = Path(csv_path).expanduser().resolve()
    candidates = [
        csv_path.parent / "camera_frames",
        csv_path.parent.parent / "camera_frames",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not auto-detect camera_frames folder. "
        "Pass --images-dir explicitly."
    )


def resolve_focus_image_path(images_dir: str | Path, filename_value: str) -> Path | None:
    """
    Resolve an image filename/path value from the CSV to a readable local path.
    """
    filename_value = filename_value.strip()
    if not filename_value:
        return None

    images_dir = Path(images_dir)
    raw_path = Path(filename_value)
    if raw_path.is_absolute() and raw_path.is_file():
        return raw_path

    candidate = images_dir / raw_path
    if candidate.is_file():
        return candidate

    by_name = images_dir / raw_path.name
    if by_name.is_file():
        return by_name

    return None


def compute_laplacian_variance(image_path: str | Path, median_ksize: int = 5) -> float:
    """
    Compute focus score as variance of the Laplacian after median blur.
    """
    _validate_odd_kernel("median_ksize", median_ksize)

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    filtered = cv2.medianBlur(image, median_ksize) if median_ksize > 1 else image
    laplacian = cv2.Laplacian(filtered, cv2.CV_64F)
    return float(laplacian.var())


def compute_focus_scores(
    image_path: str | Path,
    median_ksize: int = 5,
    log_gaussian_ksize: int = 5,
    sobel_ksize: int = 3,
    gabor_ksize: int = 21,
    gabor_sigma: float = 5.0,
    gabor_lambda: float = 10.0,
    gabor_gamma: float = 0.5,
    gabor_thetas: tuple[float, ...] = DEFAULT_GABOR_THETAS,
) -> dict[str, float]:
    """
    Compute multiple sharpness scores from one image:
      - Laplacian variance
      - LoG variance (Laplacian of Gaussian-smoothed image)
      - Sobel score (mean gradient energy / Tenengrad-style)
      - Gabor score (mean response energy over multiple orientations)
    """
    _validate_odd_kernel("median_ksize", median_ksize)
    _validate_odd_kernel("log_gaussian_ksize", log_gaussian_ksize)
    _validate_odd_kernel("sobel_ksize", sobel_ksize)
    _validate_odd_kernel("gabor_ksize", gabor_ksize)
    if gabor_sigma <= 0.0:
        raise ValueError("gabor_sigma must be > 0.")
    if gabor_lambda <= 0.0:
        raise ValueError("gabor_lambda must be > 0.")
    if gabor_gamma <= 0.0:
        raise ValueError("gabor_gamma must be > 0.")
    if not gabor_thetas:
        raise ValueError("gabor_thetas must contain at least one orientation.")

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    filtered = cv2.medianBlur(image, median_ksize) if median_ksize > 1 else image

    laplacian = cv2.Laplacian(filtered, cv2.CV_64F)
    laplacian_variance = float(laplacian.var())

    gaussian_smoothed = cv2.GaussianBlur(
        filtered,
        (log_gaussian_ksize, log_gaussian_ksize),
        0.0,
    )
    log_response = cv2.Laplacian(gaussian_smoothed, cv2.CV_64F)
    log_variance = float(log_response.var())

    sobel_x = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobel_y = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    sobel_score = float(np.mean(sobel_x * sobel_x + sobel_y * sobel_y))

    filtered_float = filtered.astype(np.float64, copy=False)
    gabor_energies: list[float] = []
    for theta in gabor_thetas:
        kernel = cv2.getGaborKernel(
            (gabor_ksize, gabor_ksize),
            gabor_sigma,
            float(theta),
            gabor_lambda,
            gabor_gamma,
            0.0,
            ktype=cv2.CV_64F,
        )
        response = cv2.filter2D(filtered_float, cv2.CV_64F, kernel)
        gabor_energies.append(float(np.mean(response * response)))
    gabor_score = float(np.mean(gabor_energies))

    return {
        "laplacian_variance": laplacian_variance,
        "log_variance": log_variance,
        "sobel_score": sobel_score,
        "gabor_score": gabor_score,
    }


def load_focus_points_from_final_csv(
    csv_path: str | Path,
    images_dir_arg: str | None = None,
    requested_position_column: str | None = None,
    median_ksize: int = 5,
    max_rows: int | None = None,
) -> tuple[str, Path, list[dict[str, str | float]]]:
    """
    Read final dataset CSV rows and compute focus scores per listed image.
    Returns:
      - resolved position column name
      - resolved image directory
      - points list with filename/image_path/position_value and score fields:
        laplacian_variance, log_variance, sobel_score, gabor_score
    """
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be a positive integer.")

    position_col = detect_focus_position_column(
        csv_path=csv_path,
        requested_column=requested_position_column,
    )
    images_dir = resolve_focus_images_dir(csv_path=csv_path, images_dir_arg=images_dir_arg)

    points: list[dict[str, str | float]] = []
    skipped_missing_file = 0
    skipped_bad_position = 0
    skipped_unreadable = 0

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=1):
            if max_rows is not None and row_index > max_rows:
                break

            filename_value = (row.get("filename") or "").strip()
            image_path = resolve_focus_image_path(images_dir, filename_value)
            if image_path is None:
                skipped_missing_file += 1
                continue

            position_value_raw = (row.get(position_col) or "").strip()
            try:
                position_value = float(position_value_raw)
            except ValueError:
                skipped_bad_position += 1
                continue

            try:
                scores = compute_focus_scores(image_path, median_ksize=median_ksize)
            except Exception:
                skipped_unreadable += 1
                continue

            points.append(
                {
                    "filename": filename_value,
                    "image_path": str(image_path),
                    "position_value": position_value,
                    "laplacian_variance": scores["laplacian_variance"],
                    "log_variance": scores["log_variance"],
                    "sobel_score": scores["sobel_score"],
                    "gabor_score": scores["gabor_score"],
                }
            )

            if len(points) % 250 == 0:
                print(f"Processed {len(points)} usable rows...")

    print(
        "Summary: "
        f"usable={len(points)}, "
        f"skipped_missing_file={skipped_missing_file}, "
        f"skipped_bad_position={skipped_bad_position}, "
        f"skipped_unreadable={skipped_unreadable}"
    )
    return position_col, images_dir, points




def main():
    # Hardcode your base directory path here
    # base_dir = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\BoPipetteData\2025_12_18-10_58"# <-- update this path
    # base_dir = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\WaynesBoroPipetteData\2026_01_19-19_44"\
    # base_dir = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\3DPrelimData\combined"
    # base_dir = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\3DPrelimData\2026_01_30-16_44" #1
    # base_dir = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\3DPrelimData\2026_01_30-16_47" #2
    # base_dir = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\3DPrelimData\2026_01_30-17_48" #3
    # base_dir = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\3DPrelimData\2026_01_30-17_59"
    # base_dir = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\3DPrelimData\2026_01_30-18_09"
    # base_dir =r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\3DPrelimData\2026_01_30-19_28"
    # base_dir = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\3DPrelimData\2026_02_02-13_28"
    # base_dir = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\3DPrelimData\2026_02_02-13_37"
    base_dir = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\3DPrelimData\2026_02_09-16_13"
    calibration_path = (
        r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\Calibration_data"
        r"\2025_11_10-13_19\calibration.json"
    )
    crop_size = 256
    camera_frames_dir = None
    for folder_name in ("camera_frames", "P_DET_IMAGES"):
        candidate_dir = os.path.join(base_dir, folder_name)
        if os.path.exists(candidate_dir):
            camera_frames_dir = candidate_dir
            break
    movement_file_path = os.path.join(base_dir, "movement_recording.csv")
    cropped_camera_frames_dir = os.path.join(base_dir, "cropped_camera_frames")
    
    # Check for required folders and files
    if camera_frames_dir is None:
        print(f"Camera frames directory not found in: {os.path.join(base_dir, 'camera_frames')} or {os.path.join(base_dir, 'P_DET_IMAGES')}")
        return
    if not os.path.exists(movement_file_path):
        print(f"Movement data file not found: {movement_file_path}")
        return
    
    # Load movement data
    movement_data = load_movement_data(movement_file_path)
    if not movement_data:
        print("No movement data loaded.")
        return
    try:
        manip_matrix_xy = load_calibration_matrix_xy(calibration_path)
    except Exception as exc:
        print(f"Failed to load calibration matrix: {exc}")
        return

    # Get list of image files (adjust extensions as needed)
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    image_files = [f for f in os.listdir(camera_frames_dir) if f.lower().endswith(image_extensions)]
    if not image_files:
        print(f"No image files found in {camera_frames_dir}")
        return
    os.makedirs(cropped_camera_frames_dir, exist_ok=True)
    
    # Extract timestamps from filenames and sort images by timestamp
    image_files_with_timestamp = []
    for img in image_files:
        _, timestamp = extract_image_data(img)
        image_files_with_timestamp.append((img, timestamp))
    image_files_with_timestamp.sort(key=lambda x: x[1])
    
    # Create CSV file in the camera_frames folder
    output_csv_path = os.path.join(camera_frames_dir, "pipette_z_data.csv")
    cropped_written = 0
    cropped_skipped = 0
    skip_reasons = {}
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "filename",
                "defocus_microns",
                "pipette_x_microns",
                "pipette_y_microns",
                "pipette_z_microns",
            ]
        )
        
        # For each image, find the movement record with the closest timestamp
        for img, timestamp in image_files_with_timestamp:
            movement_record = find_closest_movement_record(timestamp, movement_data)
            pipette_x = float(movement_record['pipette'][0])
            pipette_y = float(movement_record['pipette'][1])
            # pipette[2] holds the z-axis (defocus) value
            pipette_z = float(movement_record['pipette'][2])
            writer.writerow([img, pipette_z, pipette_x, pipette_y, pipette_z])

            source_image_path = os.path.join(camera_frames_dir, img)
            cropped_output_path = os.path.join(cropped_camera_frames_dir, img)
            zeroed_pipette_xyz = np.asarray([pipette_x, pipette_y, pipette_z], dtype=np.float64)
            success, reason = crop_tip_roi_256(
                image_path=source_image_path,
                output_path=cropped_output_path,
                manip_matrix_xy=manip_matrix_xy,
                zeroed_pipette_xyz=zeroed_pipette_xyz,
                crop_size=crop_size,
            )
            if success:
                cropped_written += 1
            else:
                cropped_skipped += 1
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    print(f"CSV file created: {output_csv_path}")
    print(
        f"Cropped images written: {cropped_written}, skipped: {cropped_skipped}, "
        f"output folder: {cropped_camera_frames_dir}"
    )
    if skip_reasons:
        print(f"Crop skip reasons: {skip_reasons}")

if __name__ == "__main__":
    main()
