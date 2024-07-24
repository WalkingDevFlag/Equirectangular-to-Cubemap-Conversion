# file_utils.py

import os

def create_output_directory(base_dir: str, dir_name: str) -> str:
    """
    Create the output directory in the specified base directory.
    """
    output_dir = os.path.join(base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_image_files(input_dir: str) -> list:
    """
    Get a list of image files from the specified directory.
    """
    return [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
