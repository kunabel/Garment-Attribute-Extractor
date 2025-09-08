import requests
import numpy as np
import cv2

def read_image(url: str) -> np.ndarray:
    """
    Downloads an image from a URL and decodes it into an OpenCV (numpy) image.

    Args:
        url (str): The URL pointing to the image.

    Returns:
        np.ndarray: Image in BGR format (as returned by OpenCV).
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # raise an exception for bad status codes

        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image from URL")

        return img

    except Exception as e:
        print(f"Error reading image from {url}: {e}")
        return None