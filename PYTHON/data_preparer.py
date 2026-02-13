# data preparer
import os
import csv

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
    camera_frames_dir = None
    for folder_name in ("camera_frames", "P_DET_IMAGES"):
        candidate_dir = os.path.join(base_dir, folder_name)
        if os.path.exists(candidate_dir):
            camera_frames_dir = candidate_dir
            break
    movement_file_path = os.path.join(base_dir, "movement_recording.csv")
    
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

    # Get list of image files (adjust extensions as needed)
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    image_files = [f for f in os.listdir(camera_frames_dir) if f.lower().endswith(image_extensions)]
    if not image_files:
        print(f"No image files found in {camera_frames_dir}")
        return
    
    # Extract timestamps from filenames and sort images by timestamp
    image_files_with_timestamp = []
    for img in image_files:
        _, timestamp = extract_image_data(img)
        image_files_with_timestamp.append((img, timestamp))
    image_files_with_timestamp.sort(key=lambda x: x[1])
    
    # Create CSV file in the camera_frames folder
    output_csv_path = os.path.join(camera_frames_dir, "pipette_z_data.csv")
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
            pipette_z = movement_record['pipette'][2]
            writer.writerow([img, pipette_z, pipette_x, pipette_y, pipette_z])
    
    print(f"CSV file created: {output_csv_path}")

if __name__ == "__main__":
    main()
