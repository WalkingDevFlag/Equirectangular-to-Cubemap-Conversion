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

def process_directory(input_dir: str, output_dir: str, face_size: int) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Processing {file_path}...")
            
            equirec_img = cv2.imread(file_path)
            if equirec_img is None:
                print(f"Error: Could not read the image at {file_path}.")
                continue
            
            gps_data = extract_gps_data(file_path)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            cubemap_faces = equirec_to_cubemap(equirec_img, face_size)
            
            save_cubemap_faces(cubemap_faces, output_dir, base_name, gps_data)

    print(f"All images processed and cubemap faces saved to {output_dir}.")

if __name__ == "__main__":
    input_directory = "E:/Random Python Scripts/Equirectangular and Cubemap/Dataset"
    output_directory = "E:/Random Python Scripts/Equirectangular and Cubemap/CubeMaps"
    cubemap_face_size = 512  # Example size, adjust as needed

    process_directory(input_directory, output_directory, cubemap_face_size)
