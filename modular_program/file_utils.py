import os

def create_output_directory(base_path: str, folder_name: str) -> str:
    """
    Create an output directory if it does not exist.
    """
    output_dir = os.path.join(base_path, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_image_files(directory: str) -> list:
    """
    Get a list of image files in the specified directory.
    """
    return [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
