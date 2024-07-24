import cv2
import numpy as np
import os
from PIL import Image
import piexif

def sample_pixels(begin: float, end: float, n: int) -> np.ndarray:
    """
    Sample pixel coordinates between 'begin' and 'end' for 'n' pixels.
    """
    x_coords = 0.5 + np.arange(n, dtype=np.float32)
    return begin + (end - begin) * x_coords / n

def equirec_to_cubemap(equirec_img: np.ndarray, out_size: int) -> list:
    """
    Convert an equirectangular image to cubemap faces.
    """
    height, width = equirec_img.shape[:2]

    u, v = np.meshgrid(sample_pixels(-1, 1, out_size),
                       sample_pixels(-1, 1, out_size),
                       indexing="ij")
    ones = np.ones((out_size, out_size), dtype=np.float32)

    list_xyz = [
        (v, u, ones),    # FRONT
        (ones, u, -v),   # RIGHT
        (-v, u, -ones),  # BACK
        (-ones, u, v),   # LEFT
        (v, -ones, u),   # UP
        (v, ones, -u)    # DOWN
    ]

    faces = []
    r = np.sqrt(u**2 + v**2 + 1)  # Same values for each face
    for x, y, z in list_xyz:
        phi = np.arcsin(y/r)
        theta = np.arctan2(x, z)

        phi_map = (phi / np.pi + 0.5) * height
        theta_map = (theta / (2 * np.pi) + 0.5) * width
    
        theta_map -= 0.5
        phi_map -= 0.5
        
        faces.append(cv2.remap(equirec_img, theta_map, phi_map, cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_WRAP))
    return faces

def extract_gps_data(image_path: str) -> dict:
    """
    Extract GPS data from the image's EXIF information.
    """
    img = Image.open(image_path)
    exif_data = piexif.load(img.info.get('exif', b''))
    gps_data = exif_data.get('GPS', {})
    return gps_data

def save_cubemap_faces(faces: list, base_name: str, output_dir: str, gps_data: dict, face_names: list) -> None:
    """
    Save each cubemap face with optional GPS data.
    """
    for face_img, face_name in zip(faces, face_names):
        face_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        exif_dict = {"GPS": gps_data} if gps_data else {}
        exif_bytes = piexif.dump(exif_dict)
        
        # Construct the output file path
        output_path = os.path.join(output_dir, f"{base_name}_{face_name}.jpg")
        
        # Save the cubemap face image
        face_img.save(output_path, "JPEG", exif=exif_bytes)
        
        # Print message to indicate saving
        print(f"Saved {face_name} face for {base_name} to {output_path}")

def convert_equirec_to_cubemap(equirec_path: str, output_dir: str, face_size: int, save_faces: dict) -> None:
    """
    Convert a directory of equirectangular images to cubemap faces and save them.
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
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define face names
    face_names = ["front", "right", "back", "left", "up", "down"]

    # Save each cubemap face image with GPS data if available
    gps_data = extract_gps_data(equirec_path)
    save_cubemap_faces(cubemap_faces, base_name, output_dir, gps_data, [face_name for face_name, save in save_faces.items() if save])
    
def main():
    # Path to the directory containing equirectangular images
    input_directory = input("Enter the path to the directory containing equirectangular images: ").strip()

    # Create the output directory named "CubeMaps" in the same directory as the script
    output_directory = os.path.join(os.path.dirname(__file__), "CubeMaps")

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
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_directory, filename)
            print(f"Processing image: {image_path}")
            convert_equirec_to_cubemap(image_path, output_directory, cubemap_face_size, faces_to_save)

    print("All images have been processed.")

if __name__ == "__main__":
    main()
