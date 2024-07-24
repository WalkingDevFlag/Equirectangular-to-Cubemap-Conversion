# main.py

import cv2
import os
from image_utils import equirec_to_cubemap, extract_gps_data, save_cubemap_faces
from file_utils import create_output_directory, get_image_files

def convert_equirec_to_cubemap(equirec_path: str, output_dir: str, face_size: int, save_faces: dict) -> None:
    """
    Convert an equirectangular image to cubemap faces and save them.
    """
    # Load the equirectangular image
    equirec_img = cv2.imread(equirec_path)
    if equirec_img is None:
        print(f"Error: Could not read the image at {equirec_path}.")
        return

    # Extract base name without extension
    base_name = os.path.splitext(os.path.basename(equirec_path))[0]

    # Convert equirectangular image to cubemap faces
    cubemap_faces = equirec_to_cubemap(equirec_img, face_size)
    
    # Save each cubemap face image with GPS data if available
    gps_data = extract_gps_data(equirec_path)
    face_names = [name for name, save in save_faces.items() if save]
    save_cubemap_faces(cubemap_faces, base_name, output_dir, gps_data, face_names)

def main():
    # Path to the directory containing equirectangular images
    input_directory = input("Enter the path to the directory containing equirectangular images: ").strip()

    # Create the output directory named "CubeMaps" in the same directory as the script
    output_directory = create_output_directory(os.path.dirname(__file__), "CubeMaps")

    # Ask user for the cubemap face size
    cubemap_face_size = int(input("Enter the size for cubemap faces (e.g., 512): ").strip())

    # Check which faces to save
    faces_to_save = {
        "front": input("Save Front face? (y/n): ").strip().lower() in ['y', 'yes'],
        "right": input("Save Right face? (y/n): ").strip().lower() in ['y', 'yes'],
        "back": input("Save Back face? (y/n): ").strip().lower() in ['y', 'yes'],
        "left": input("Save Left face? (y/n): ").strip().lower() in ['y', 'yes'],
        "up": input("Save Up face? (y/n): ").strip().lower() in ['y', 'yes'],
        "down": input("Save Down face? (y/n): ").strip().lower() in ['y', 'yes']
    }

    # Process each image in the directory
    image_files = get_image_files(input_directory)
    for filename in image_files:
        image_path = os.path.join(input_directory, filename)
        print(f"Processing image: {image_path}")
        convert_equirec_to_cubemap(image_path, output_directory, cubemap_face_size, faces_to_save)

    print("All images have been processed.")

if __name__ == "__main__":
    main()
