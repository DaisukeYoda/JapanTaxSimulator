import os
import shutil

SOURCE_FILE = "tax_analysis_report_japanese.txt"
DESTINATION_DIR = "results/"
DESTINATION_FILE = os.path.join(DESTINATION_DIR, "research_report_tax_analysis_effects_jp.txt") # More descriptive name

# Ensure destination directory exists (it should, but good practice)
if not os.path.exists(DESTINATION_DIR):
    os.makedirs(DESTINATION_DIR)
    print(f"Created directory: {DESTINATION_DIR}")

if os.path.exists(SOURCE_FILE):
    # Check if DESTINATION_FILE already exists, and remove it if it does, to avoid shutil.Error
    if os.path.exists(DESTINATION_FILE):
        print(f"Destination file '{DESTINATION_FILE}' already exists. Removing it first.")
        try:
            os.remove(DESTINATION_FILE)
            print(f"Successfully removed existing '{DESTINATION_FILE}'.")
        except OSError as e:
            print(f"Error removing existing destination file '{DESTINATION_FILE}': {e}")
            # Depending on requirements, might raise error or try to proceed
            # For now, let's raise to be safe if it cannot be removed
            raise

    shutil.move(SOURCE_FILE, DESTINATION_FILE)
    print(f"Moved '{SOURCE_FILE}' to '{DESTINATION_FILE}'")
else:
    print(f"Error: Source file '{SOURCE_FILE}' not found in project root.")
    # If the file is not found, this subtask will indicate a problem.
    # This could happen if the previous subtask didn't place it as expected.
    # Raising an exception will make the subtask fail clearly.
    raise FileNotFoundError(f"Source file {SOURCE_FILE} not found. Report generation might have failed or placed it elsewhere.")
