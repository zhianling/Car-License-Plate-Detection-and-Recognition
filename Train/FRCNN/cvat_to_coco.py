import os
import xml.etree.ElementTree as ET
import json
import cv2
import sys
import time
import concurrent.futures
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
BASE_DATA_DIR = "../../data"
CVAT_ANNOTATION_DIR = os.path.join(BASE_DATA_DIR, "annotation", "cvat")
VIDEO_DIR = os.path.join(BASE_DATA_DIR, "video")
COCO_ANNOTATION_DIR = os.path.join(BASE_DATA_DIR, "annotation", "coco")
FRAME_OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "frame")
START_NUM = 1
END_NUM = 58 # Adjust if your numbering is different
NUM_WORKERS = 24 # Number of threads to use
FRAME_FILENAME_FORMAT = "frame_{:06d}.jpg" # e.g., frame_000000.jpg
# --- End Configuration ---

# Make paths absolute relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
BASE_DATA_DIR = os.path.join(script_dir, BASE_DATA_DIR) # Make paths relative to script
CVAT_ANNOTATION_DIR = os.path.join(script_dir, CVAT_ANNOTATION_DIR)
VIDEO_DIR = os.path.join(script_dir, VIDEO_DIR)
COCO_ANNOTATION_DIR = os.path.join(script_dir, COCO_ANNOTATION_DIR)
FRAME_OUTPUT_DIR = os.path.join(script_dir, FRAME_OUTPUT_DIR)


def parse_cvat_xml(xml_path, file_id_str):
    """
    Parses a CVAT 1.1 XML file (interpolation format) and extracts
    annotated frames and their annotations. (Added file_id_str for logging)

    Args:
        xml_path (str): Path to the CVAT XML file.
        file_id_str(str): Identifier string (e.g., "01") for logging.

    Returns:
        tuple: (categories, annotations_by_frame, annotated_frame_indices, original_size)
               categories (dict): {name: id} mapping.
               annotations_by_frame (dict): {frame_idx: [list of annotations]}
               annotated_frame_indices (set): Set of frame indices with annotations.
               original_size (dict): {'width': w, 'height': h} or None if not found.
        Returns None if parsing fails or file not found.
    """
    if not os.path.exists(xml_path):
        print(f"[{file_id_str}] Error: XML file not found: {xml_path}")
        return None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # --- Extract Categories (Labels) ---
        categories = {}
        category_id_counter = 1 # COCO category IDs usually start from 1
        labels_element = root.find(".//job/labels") # More specific path
        if labels_element is None:
             labels_element = root.find(".//meta/task/labels") # Alternative path
        if labels_element is not None:
            for label in labels_element.findall("label"):
                name = label.findtext("name")
                if name and name not in categories:
                    categories[name] = category_id_counter
                    category_id_counter += 1
        else:
             print(f"[{file_id_str}] Warning: Could not find <labels> section in {xml_path}")
             # Try finding labels from tracks if meta is missing/incomplete
             for track in root.findall("track"):
                 label = track.get("label")
                 if label and label not in categories:
                     categories[label] = category_id_counter
                     category_id_counter += 1

        if not categories:
            print(f"[{file_id_str}] Error: No labels found in {xml_path}. Cannot proceed.")
            return None

        # --- Extract Original Size ---
        original_size = None
        size_element = root.find(".//original_size") # More specific path
        if size_element is None:
             size_element = root.find(".//meta/task/original_size") # Alternative
        if size_element is not None:
            width = size_element.findtext("width")
            height = size_element.findtext("height")
            if width and height:
                original_size = {"width": int(width), "height": int(height)}
        else:
             print(f"[{file_id_str}] Warning: Could not find <original_size> in {xml_path}")

        # --- Extract Annotations and Identify Annotated Frames ---
        annotations_by_frame = defaultdict(list)
        annotated_frame_indices = set()
        annotation_id_counter = 1

        for track in root.findall("track"):
            track_id = track.get("id")
            label = track.get("label")
            if label not in categories:
                print(f"[{file_id_str}] Warning: Label '{label}' in track {track_id} not found in meta. Skipping track.")
                continue
            category_id = categories[label]

            for box in track.findall("box"):
                frame_str = box.get("frame")
                if frame_str is None:
                    continue
                frame_idx = int(frame_str)
                annotated_frame_indices.add(frame_idx)

                xtl = float(box.get("xtl"))
                ytl = float(box.get("ytl"))
                xbr = float(box.get("xbr"))
                ybr = float(box.get("ybr"))
                x_min = xtl
                y_min = ytl
                width = xbr - xtl
                height = ybr - ytl
                coco_bbox = [x_min, y_min, width, height]
                area = width * height

                annotation = {
                    "id": annotation_id_counter, # Temporary ID, will be re-indexed later
                    "image_id": frame_idx, # Will link to COCO image ID later
                    "category_id": category_id,
                    "bbox": coco_bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": [],
                    "attributes": {}
                }
                annotations_by_frame[frame_idx].append(annotation)
                annotation_id_counter += 1 # Note: this counter is local to this XML parsing

        return categories, annotations_by_frame, sorted(list(annotated_frame_indices)), original_size

    except ET.ParseError as e:
        print(f"[{file_id_str}] Error parsing XML file {xml_path}: {e}")
        return None
    except Exception as e:
        print(f"[{file_id_str}] An unexpected error occurred while parsing {xml_path}: {e}")
        return None


def create_coco_json(categories, annotations_by_frame, annotated_frame_indices, original_size, file_id_str):
    """
    Creates the COCO JSON structure from parsed CVAT data, including only annotated frames.
    """
    coco_output = {
        "info": {
            "description": f"Dataset {file_id_str} converted from CVAT XML",
            "version": "1.0", "year": time.localtime().tm_year,
            "contributor": "Conversion Script", "date_created": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [], "images": [], "annotations": [], "categories": []
    }

    for name, id_val in categories.items():
        coco_output["categories"].append({"id": id_val, "name": name, "supercategory": "object"})

    if not annotated_frame_indices:
         print(f"[{file_id_str}] Warning: No annotated frames found. COCO file will be sparse.")
         return coco_output

    img_width, img_height = 1, 1 # Default/placeholder
    if original_size:
        img_width, img_height = original_size['width'], original_size['height']
    else:
        print(f"[{file_id_str}] Warning: Original size unknown for COCO. Setting dummy values (1x1).")

    image_id_map = {frame_idx: frame_idx for frame_idx in annotated_frame_indices}

    for frame_idx in annotated_frame_indices:
        coco_image_id = image_id_map[frame_idx]
        frame_filename = FRAME_FILENAME_FORMAT.format(frame_idx)
        relative_frame_path = os.path.join(file_id_str, frame_filename)

        coco_output["images"].append({
            "id": coco_image_id, "width": img_width, "height": img_height,
            "file_name": relative_frame_path, "license": 0, "flickr_url": "",
            "coco_url": "", "date_captured": ""
        })

        for ann in annotations_by_frame.get(frame_idx, []):
            ann_copy = ann.copy() # Work on a copy to avoid modifying the original dict if reused
            ann_copy["image_id"] = coco_image_id
            coco_output["annotations"].append(ann_copy)

    # Re-index annotation IDs sequentially for this specific JSON file
    for i, ann in enumerate(coco_output["annotations"]):
        ann['id'] = i + 1

    return coco_output

def extract_annotated_frames(video_path, annotated_frame_indices, output_dir_base, file_id_str):
    """
    Extracts specific frames from a video file. (Modified output_dir handling)

    Args:
        video_path (str): Path to the video file.
        annotated_frame_indices (list/set): Indices of frames to extract.
        output_dir_base (str): Base directory (e.g., data/frame). Subdir will be created.
        file_id_str (str): The identifier (e.g., "01") to create a subdirectory.

    Returns:
        tuple: (success_flag, actual_size_dict or None)
               bool: True if video opened and extraction attempted.
               dict: Actual width/height found from video {'width': w, 'height': h} or None
    """
    if not os.path.exists(video_path):
        print(f"[{file_id_str}] Error: Video file not found: {video_path}")
        return False, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[{file_id_str}] Error: Could not open video file: {video_path}")
        return False, None

    # Create subdirectory for this video's frames
    specific_frame_dir = os.path.join(output_dir_base, file_id_str)
    try:
        os.makedirs(specific_frame_dir, exist_ok=True)
    except OSError as e:
         print(f"[{file_id_str}] Error creating directory {specific_frame_dir}: {e}")
         cap.release()
         return False, None # Cannot proceed if output dir fails


    frame_count_prop = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_size = {"width": vid_width, "height": vid_height}
    frame_count = int(frame_count_prop) if frame_count_prop > 0 else "unknown"
    # print(f"[{file_id_str}] Video Info: {frame_count} frames, {vid_width}x{vid_height}")

    frames_to_extract_set = set(annotated_frame_indices)
    extracted_count = 0
    current_frame_idx = 0
    frames_written = 0
    frames_failed_write = 0

    while True:
        # Optimization: Check if we still need to read frames
        if extracted_count == len(frames_to_extract_set) and len(frames_to_extract_set) > 0:
             # print(f"[{file_id_str}] All {extracted_count} annotated frames found. Stopping read.")
             break

        ret, frame = cap.read()
        if not ret:
            # print(f"[{file_id_str}] Reached end of video stream at frame {current_frame_idx}.")
            break # End of video

        if current_frame_idx in frames_to_extract_set:
            extracted_count += 1 # Mark as found, even if write fails
            frame_filename = FRAME_FILENAME_FORMAT.format(current_frame_idx)
            output_path = os.path.join(specific_frame_dir, frame_filename)
            try:
                # Only write if it doesn't exist (useful for reruns, though less efficient)
                # if not os.path.exists(output_path):
                cv2.imwrite(output_path, frame)
                frames_written += 1
            except Exception as e:
                 print(f"[{file_id_str}] Error writing frame {current_frame_idx} to {output_path}: {e}")
                 frames_failed_write += 1

        current_frame_idx += 1

    cap.release()
    # Final report for this ID
    print(f"[{file_id_str}] Frame Extraction Summary: Found={extracted_count}/{len(frames_to_extract_set)}, Written={frames_written}, WriteErrors={frames_failed_write}")

    if extracted_count != len(frames_to_extract_set):
        print(f"[{file_id_str}] Warning: Mismatch in expected frames. Expected {len(frames_to_extract_set)}, found {extracted_count}. Video might be shorter or frames unreadable.")

    # Return True as we attempted extraction, along with the video size
    return True, actual_size

def process_single_id(file_id_str):
    """
    Worker function to process one ID (XML parse, frame extract, COCO generate).
    Uses global config variables for paths.
    """
    print(f"[{file_id_str}] Starting processing...")
    xml_path = os.path.join(CVAT_ANNOTATION_DIR, f"{file_id_str}.xml")
    video_path = os.path.join(VIDEO_DIR, f"{file_id_str}.mp4")
    coco_json_path = os.path.join(COCO_ANNOTATION_DIR, f"{file_id_str}.json")

    # 1. Parse CVAT XML
    parse_result = parse_cvat_xml(xml_path, file_id_str)
    if parse_result is None:
        return "error", file_id_str, "XML parsing failed"

    categories, annotations_by_frame, annotated_frame_indices, original_size = parse_result

    if not annotated_frame_indices:
        print(f"[{file_id_str}] No annotations found. Skipping frame extraction and COCO generation.")
        # Optionally create empty COCO json here if needed
        return "skipped", file_id_str, "No annotations"

    # 2. Extract Annotated Frames
    extract_success, actual_video_size = extract_annotated_frames(
        video_path,
        annotated_frame_indices,
        FRAME_OUTPUT_DIR, # Base output dir
        file_id_str
    )

    if not extract_success:
        # Error messages are printed inside extract_annotated_frames
        return "error", file_id_str, "Frame extraction failed (video open/read error)"

    # Update original_size if needed (using actual video dimensions is preferred)
    if original_size is None and actual_video_size:
        print(f"[{file_id_str}] Using actual video dimensions ({actual_video_size['width']}x{actual_video_size['height']}) for COCO metadata.")
        original_size = actual_video_size
    elif original_size and actual_video_size and (original_size['width'] != actual_video_size['width'] or original_size['height'] != actual_video_size['height']):
        print(f"[{file_id_str}] Warning: XML size ({original_size['width']}x{original_size['height']}) differs from video ({actual_video_size['width']}x{actual_video_size['height']}). Using video size.")
        original_size = actual_video_size
    elif original_size is None and actual_video_size is None:
         print(f"[{file_id_str}] Error: Cannot determine image dimensions from XML or Video.")
         return "error", file_id_str, "Cannot determine image dimensions"


    # 3. Create and Save COCO JSON
    coco_data = create_coco_json(
        categories,
        annotations_by_frame,
        annotated_frame_indices,
        original_size,
        file_id_str
    )

    try:
        # Ensure the coco annotation directory exists (thread-safe check)
        os.makedirs(COCO_ANNOTATION_DIR, exist_ok=True)
        with open(coco_json_path, 'w') as f:
            json.dump(coco_data, f, indent=4)
        # print(f"[{file_id_str}] Saved COCO JSON to {coco_json_path}") # Can be noisy
    except Exception as e:
        print(f"[{file_id_str}] Error writing COCO JSON file {coco_json_path}: {e}")
        return "error", file_id_str, "COCO JSON writing failed"

    print(f"[{file_id_str}] Processing finished successfully.")
    return "success", file_id_str, "Completed"


# --- Main Execution ---
def main():
    """Main function to orchestrate the conversion and extraction using ThreadPoolExecutor."""
    start_time = time.time()
    print("Starting CVAT to COCO (Annotated Only) Conversion & Frame Extraction (Threaded)...")
    print(f"Using {NUM_WORKERS} worker threads.")
    print(f"CVAT XML Source: {CVAT_ANNOTATION_DIR}")
    print(f"Video Source: {VIDEO_DIR}")
    print(f"COCO JSON Output: {COCO_ANNOTATION_DIR}")
    print(f"Frame Output: {FRAME_OUTPUT_DIR}")

    # --- Pre-checks ---
    if not os.path.isdir(CVAT_ANNOTATION_DIR) or not os.path.isdir(VIDEO_DIR):
         print("\nError: Input directories not found!")
         print(f"Looked for CVAT annotations in: {CVAT_ANNOTATION_DIR}")
         print(f"Looked for videos in: {VIDEO_DIR}")
         sys.exit(1)

    # Ensure base output directories exist (subdirs created by workers)
    os.makedirs(COCO_ANNOTATION_DIR, exist_ok=True)
    os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)

    processed_count = 0
    skipped_count = 0
    error_count = 0
    futures = []
    ids_to_process = [f"{i:02d}" for i in range(START_NUM, END_NUM + 1)]

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        for file_id_str in ids_to_process:
            future = executor.submit(process_single_id, file_id_str)
            futures.append(future)

        print(f"\nSubmitted {len(futures)} tasks to the thread pool. Waiting for completion...")

        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                status, file_id_str, message = future.result()
                if status == "success":
                    processed_count += 1
                elif status == "skipped":
                    skipped_count += 1
                    # print(f"[*] ID {file_id_str} skipped: {message}") # Message printed in worker
                elif status == "error":
                    error_count += 1
                    # print(f"[!] Error processing ID {file_id_str}: {message}") # Message printed in worker
            except Exception as exc:
                # Catch exceptions raised *within* the worker function that weren't handled
                # Need to find out which file_id this was if possible, but difficult directly from future
                error_count += 1
                print(f"[!!!] An unexpected exception occurred in a worker thread: {exc}")


    # --- Summary ---
    end_time = time.time()
    duration = end_time - start_time
    print("\n--- Processing Summary ---")
    print(f"Total time taken: {duration:.2f} seconds")
    print(f"Successfully processed: {processed_count} IDs")
    print(f"Skipped (no annotations): {skipped_count} IDs")
    print(f"Encountered errors: {error_count} IDs")
    total_expected = END_NUM - START_NUM + 1
    if processed_count + skipped_count + error_count != total_expected:
        print(f"Warning: Counts do not sum up to the expected total of {total_expected} IDs.")
    print("------------------------")


if __name__ == "__main__":
    main()