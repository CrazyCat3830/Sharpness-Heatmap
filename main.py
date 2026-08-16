import cv2
import numpy as np
from pathlib import Path

LOW_PERCENTILE = 5
HIGH_PERCENTILE = 97.5


def sharpness_map(image, window_size=48, step=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

    map_height = (height - window_size) // step + 1
    map_width = (width - window_size) // step + 1

    sharpness = np.zeros((map_height, map_width), dtype=np.float32)

    for row in range(map_height):
        for col in range(map_width):
            y1 = row * step
            y2 = y1 + window_size

            x1 = col * step
            x2 = x1 + window_size

            patch = gray[y1:y2, x1:x2]

            laplacian = cv2.Laplacian(patch, cv2.CV_64F)
            sharpness[row, col] = laplacian.var()

    return sharpness


def normalize_sharpness(sharpness):
    low = np.percentile(sharpness, LOW_PERCENTILE)
    high = np.percentile(sharpness, HIGH_PERCENTILE)

    if high == low:
        return np.zeros_like(sharpness, dtype=np.uint8)

    normalized = (sharpness - low) / (high - low)
    normalized = np.clip(normalized, 0, 1)

    return (normalized * 255).astype(np.uint8)


def create_heatmap(image, sharpness):
    normalized = normalize_sharpness(sharpness)

    # Smooth the sharpness map before resizing
    normalized = cv2.GaussianBlur(
        normalized,
        (0, 0),
        sigmaX=2
    )

    # Resize sharpness map to original image size
    resized = cv2.resize(
        normalized,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    # Create colored heatmap
    heatmap = cv2.applyColorMap(
        resized,
        cv2.COLORMAP_JET
    )

    # Alpha depends on sharpness:
    # blurry areas stay close to the original image
    alpha = resized.astype(np.float32) / 255.0
    alpha = alpha * 0.65

    alpha = alpha[:, :, np.newaxis]

    result = (
        image.astype(np.float32) * (1 - alpha)
        + heatmap.astype(np.float32) * alpha
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    input_path = Path("input/photo.jpg")
    output_path = Path("output/sharpness_heatmap.jpg")

    image = cv2.imread(str(input_path))

    if image is None:
        raise FileNotFoundError(f"Could not open {input_path}")

    sharpness = sharpness_map(
        image,
        window_size=48,
        step=8
    )

    result = create_heatmap(image, sharpness)

    output_path.parent.mkdir(exist_ok=True)

    cv2.imwrite(str(output_path), result)

    print(f"Saved heatmap to {output_path}")


if __name__ == "__main__":
    main()
