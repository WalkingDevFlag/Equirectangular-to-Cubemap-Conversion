import cv2
import numpy as np
import os
from PIL import Image
import piexif

def sample_pixels(begin: float, end: float, n: int) -> np.ndarray:
    x_coords = 0.5 + np.arange(n, dtype=np.float32)
    return begin + (end - begin) * x_coords / n

def equirec_to_cubemap(equirec_img: np.ndarray, out_size: int) -> list:
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
    img = Image.open(image_path)
    exif_data = piexif.load(img.info.get('exif', b''))
    gps_data = exif_data.get('GPS', {})
    return gps_data

def save_cubemap_faces(faces: list, output_dir: str, base_name: str, gps_data: dict) -> None:
    face_names = ["front", "right", "back", "left", "up", "down"]
    for i, face in enumerate(faces):
        face_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        exif_dict = {"GPS": gps_data} if gps_data else {}
        exif_bytes = piexif.dump(exif_dict)
        face_path = os.path.join(output_dir, f"{base_name}_{face_names[i]}.jpg")
        face_img.save(face_path, "JPEG", exif=exif_bytes)

def convert_equirec_to_cubemap(equirec_image_path: str, face_size: int) -> None:
    # Load equirectangular image
    equirec_img = cv2.imread(equirec_image_path)
    if equirec_img is None:
        print(f"Error: Could not read the image at {equirec_image_path}.")
        return

    # Extract base name without extension
    base_name = os.path.splitext(os.path.basename(equirec_image_path))[0]

    # Convert equirectangular image to cubemap faces
    cubemap_faces = equirec_to_cubemap(equirec_img, face_size)
    
    # Define output directory
    output_dir = os.path.join(os.path.dirname(__file__), "CubeMaps")
    os.makedirs(output_dir, exist_ok=True)

    # Extract GPS data
    gps_data = extract_gps_data(equirec_image_path)
    
    # Save cubemap faces
    save_cubemap_faces(cubemap_faces, output_dir, base_name, gps_data)

    print(f"Cubemap faces saved to {output_dir}.")

if __name__ == "__main__":
    input_directory = "E:/Random Python Scripts/Equirectangular and Cubemap/Dataset"  # Adjust the path as needed
    cubemap_face_size = 512  # Example size, adjust as needed

    # Process all images in the input directory
    for file_name in os.listdir(input_directory):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            equirec_image_path = os.path.join(input_directory, file_name)
            convert_equirec_to_cubemap(equirec_image_path, cubemap_face_size)
