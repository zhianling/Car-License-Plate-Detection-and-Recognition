import os
import shutil
import sys

# --- Configuration ---
SOURCE_BASE_DIR = "cvat_datasets"
DEST_VIDEO_DIR = os.path.join("data", "video")
DEST_ANNOTATION_DIR = os.path.join("data", "annotation", "cvat")
START_NUM = 1
END_NUM = 58
# --- End Configuration ---

def main():
    """Main function to move and rename CVAT dataset files."""

    # Get the absolute path of the source directory relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    source_dir_abs = os.path.join(script_dir, SOURCE_BASE_DIR)

    # Check if the source directory exists
    if not os.path.isdir(source_dir_abs):
        print(f"Error: Source directory '{SOURCE_BASE_DIR}' not found in the script's directory or current working directory.")
        print(f"Looked for: {source_dir_abs}")
        sys.exit(1) # Exit the script with an error code

    # Create destination directories if they don't exist
    # os.makedirs creates parent directories as needed (like mkdir -p)
    # exist_ok=True prevents an error if the directories already exist
    print("Ensuring destination directories exist...")
    os.makedirs(os.path.join(script_dir, DEST_VIDEO_DIR), exist_ok=True)
    os.makedirs(os.path.join(script_dir, DEST_ANNOTATION_DIR), exist_ok=True)
    print("Destination directories ensured.")

    print("\nStarting file moving process...")
    moved_count = 0
    warning_count = 0

    # Loop through the numbered directories
    for i in range(START_NUM, END_NUM + 1):
        # Format number with leading zero (e.g., 1 -> "01", 10 -> "10")
        dir_num_str = f"{i:02d}"
        current_source_subdir = os.path.join(source_dir_abs, dir_num_str)

        if os.path.isdir(current_source_subdir):
            # Define source file paths
            source_video_path = os.path.join(current_source_subdir, "video.mp4")
            source_annotation_path = os.path.join(current_source_subdir, "annotations.xml")

            # Define destination file paths (using absolute paths for clarity with shutil.move)
            dest_video_path = os.path.join(script_dir, DEST_VIDEO_DIR, f"{dir_num_str}.mp4")
            dest_annotation_path = os.path.join(script_dir, DEST_ANNOTATION_DIR, f"{dir_num_str}.xml")

            # --- Move Video File ---
            if os.path.isfile(source_video_path):
                try:
                    print(f"Moving '{source_video_path}' to '{dest_video_path}'")
                    shutil.move(source_video_path, dest_video_path)
                    moved_count += 1
                except Exception as e:
                    print(f"Error moving video from {dir_num_str}: {e}")
                    warning_count += 1
            else:
                print(f"Warning: Video file not found in '{current_source_subdir}'")
                warning_count += 1

            # --- Move Annotation File ---
            if os.path.isfile(source_annotation_path):
                try:
                    print(f"Moving '{source_annotation_path}' to '{dest_annotation_path}'")
                    shutil.move(source_annotation_path, dest_annotation_path)
                    moved_count += 1
                except Exception as e:
                    print(f"Error moving annotation from {dir_num_str}: {e}")
                    warning_count += 1
            else:
                print(f"Warning: Annotation file not found in '{current_source_subdir}'")
                warning_count += 1
        else:
            print(f"Warning: Source subdirectory '{current_source_subdir}' not found. Skipping.")
            warning_count += 1

    print("\n-------------------------------------")
    print("File moving process completed.")
    print(f"Total files/annotations processed for moving: {moved_count}")
    print(f"Total warnings (missing files/dirs): {warning_count}")
    print(f"Videos moved to: {os.path.join(script_dir, DEST_VIDEO_DIR)}")
    print(f"Annotations moved to: {os.path.join(script_dir, DEST_ANNOTATION_DIR)}")
    print("-------------------------------------")

# Standard Python practice to run the main function when the script is executed
if __name__ == "__main__":
    main()