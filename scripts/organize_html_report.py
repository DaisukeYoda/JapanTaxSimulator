import os
import shutil

SOURCE_HTML_FILE = "tax_analysis_report_jp.html"
DESTINATION_DIR = "results/"
# The problem description implies the text report and HTML report might have the same final name.
# Let's adjust the HTML final name slightly to avoid exact collision if they were in the same dir,
# though here the text report is already in results/.
# For clarity and to match the problem's FINAL_HTML_FILENAME.
FINAL_HTML_FILENAME = "research_report_tax_analysis_effects_jp.html"
DESTINATION_HTML_FILE = os.path.join(DESTINATION_DIR, FINAL_HTML_FILENAME)

# Ensure destination directory exists
if not os.path.exists(DESTINATION_DIR):
    os.makedirs(DESTINATION_DIR)
    print(f"Created directory: {DESTINATION_DIR}")

if os.path.exists(SOURCE_HTML_FILE):
    # If the destination file already exists, remove it first to avoid errors with shutil.move on some OS
    if os.path.exists(DESTINATION_HTML_FILE):
        try:
            os.remove(DESTINATION_HTML_FILE)
            print(f"Removed existing file: {DESTINATION_HTML_FILE}")
        except OSError as e:
            print(f"Error removing existing destination file '{DESTINATION_HTML_FILE}': {e}")
            raise # Re-raise to make subtask fail if removal fails

    shutil.move(SOURCE_HTML_FILE, DESTINATION_HTML_FILE)
    print(f"Moved '{SOURCE_HTML_FILE}' to '{DESTINATION_HTML_FILE}'")

    # Verify the move
    if os.path.exists(DESTINATION_HTML_FILE) and not os.path.exists(SOURCE_HTML_FILE):
        print(f"Successfully verified that '{DESTINATION_HTML_FILE}' exists and '{SOURCE_HTML_FILE}' has been removed.")
    else:
        print(f"Error verifying the move. Check '{SOURCE_HTML_FILE}' and '{DESTINATION_HTML_FILE}'.")
        # If source is gone but dest not there, or dest is there but source also there (copy instead of move?)
        if not os.path.exists(DESTINATION_HTML_FILE):
             raise SystemError(f"File move verification failed: Destination file '{DESTINATION_HTML_FILE}' does not exist after move.")
        if os.path.exists(SOURCE_HTML_FILE):
             raise SystemError(f"File move verification failed: Source file '{SOURCE_HTML_FILE}' still exists after move.")
        # General error if logic above is insufficient
        raise SystemError("File move verification failed due to an unexpected state.")


else:
    print(f"Error: Source HTML file '{SOURCE_HTML_FILE}' not found in project root.")
    raise FileNotFoundError(f"Source HTML file '{SOURCE_HTML_FILE}' not found. HTML report generation might have failed or placed it elsewhere.")
