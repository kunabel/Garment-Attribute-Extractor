import requests
import numpy as np
import cv2

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

def read_image(url: str) -> np.ndarray:
    """
    Downloads an image from a URL and decodes it into an OpenCV (numpy) image.
    Accepts only valid images under 10 MB.

    Args:
        url (str): The URL pointing to the image.

    Returns:
        np.ndarray: Image in BGR format (as returned by OpenCV), or None if invalid.
    """
    try:
        # Stream download so we can check size before reading everything
        with requests.get(url, stream=True, timeout=10) as response:
            response.raise_for_status()

            # Check Content-Type header
            content_type = response.headers.get("Content-Type", "").lower()
            if not content_type.startswith("image/"):
                raise ValueError(f"Invalid content type: {content_type}")

            # Enforce maximum size (10MB)
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_IMAGE_SIZE:
                raise ValueError(f"Image too large: {int(content_length)/1024/1024:.2f} MB")

            # Accumulate bytes but stop if exceeding max size
            data = bytearray()
            for chunk in response.iter_content(8192):
                data.extend(chunk)
                if len(data) > MAX_IMAGE_SIZE:
                    raise ValueError("Image exceeded 10 MB during download")

        # Decode with OpenCV
        image_array = np.asarray(data, dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")

        return img

    except Exception as e:
        print(f"Error reading image from {url}: {e}")
        return None
