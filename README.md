# Equirectangular to Cubemap Conversion Script

## Overview

This script converts equirectangular images to cubemap faces and saves them with optional GPS data. It processes all images in a specified directory and allows you to selectively save different cubemap faces.

## Prerequisites

Ensure you have the following Python libraries installed:
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Pillow (`PIL`)
- Piexif (`piexif`)

You can install these using pip:
```sh
pip install opencv-python numpy pillow piexif
```

## Files Included

- **`main.py`**: Main script that orchestrates the conversion process.
- **`image_utils.py`**: Contains functions for image processing and cubemap conversion.
- **`file_utils.py`**: Contains utility functions for file handling.

## How to Run

1. **Prepare Your Environment**

   Make sure you have the required Python libraries installed. Create a directory where your equirectangular images are stored.

2. **Place the Script Files**

   Ensure the `main.py`, `image_utils.py`, and `file_utils.py` files are in the same directory.

3. **Run the Script**

   Open a terminal or command prompt and navigate to the directory containing the script files. Execute the following command:
   ```sh
   python main.py
   ```

4. **Provide Input**

   The script will prompt you to provide:
   - **Path to the directory containing equirectangular images**: Enter the full path to the directory where your images are stored.
   - **Size for cubemap faces**: Enter the size (e.g., 512 or 640) for the cubemap faces.
   - **Which faces to save**: For each cubemap face (Front, Right, Back, Left, Up, Down), enter `y` or `yes` to save that face, or `n` to skip it.

5. **Output**

   The script will create a folder named `CubeMaps` in the same directory as the script. Inside this folder, you will find the cubemap faces saved for each image according to your choices.

6. **Review Results**

   After processing, check the `CubeMaps` folder for the saved images. Each image will be named with the original image's base name followed by the face name (e.g., `pano_000001_000000_front.jpg`).

## Troubleshooting

- **Error: Could not read the image**: Ensure the image path is correct and the image file exists.
- **Missing library errors**: Ensure all required libraries are installed.

For any other issues or questions, consult the documentation of the respective libraries or seek help from online forums.
