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

def save_cubemap_faces(faces: list, output_dir: str, base_name: str, face_names: list, gps_data: dict) -> None:
    for face_img, face_name in zip(faces, face_names):
        if face_name in face_names:
            # Construct the output file path
            output_path = os.path.join(output_dir, f"{base_name}_{face_name}.jpg")
            
            # Save the cubemap face image
            cv2.imwrite(output_path, face_img)

            # If GPS data is available, copy it to the cubemap face images
            if gps_data:
                exif_dict = {"GPS": gps_data}
                exif_bytes = piexif.dump(exif_dict)
                face_img_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                face_img_pil.save(output_path, "JPEG", exif=exif_bytes)

def convert_equirec_to_cubemap(input_dir: str, face_size: int) -> None:
    output_dir = os.path.join(os.path.dirname(__file__), "CubeMaps")
    os.makedirs(output_dir, exist_ok=True)

    face_names = ["front", "right", "back", "left", "up", "down"]
    
    # Ask user which faces to save
    selected_faces = []
    for face in face_names:
        response = input(f"Save {face.capitalize()} face? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            selected_faces.append(face)
    
    if not selected_faces:
        print("No faces selected. Exiting.")
        return

    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if os.path.isfile(file_path):
            equirec_img = cv2.imread(file_path)
            if equirec_img is None:
                print(f"Error reading image {file_path}. Skipping.")
                continue

            base_name = os.path.splitext(file_name)[0]
            gps_data = extract_gps_data(file_path)

            cubemap_faces = equirec_to_cubemap(equirec_img, face_size)
            save_cubemap_faces(cubemap_faces, output_dir, base_name, selected_faces, gps_data)
    
    print(f"Cubemap faces saved to {output_dir}.")

if __name__ == "__main__":
    input_directory = "E:/Random Python Scripts/Equirectangular and Cubemap/Dataset"
    cubemap_face_size = 512  # Example size, adjust as needed

    convert_equirec_to_cubemap(input_directory, cubemap_face_size)
