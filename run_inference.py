############################################################################################################################
#   How to run inference on a single image:
#   python <image_path_or_folder> --endpoint https://app.happyrobot.ai/predict/{endpoint-name} --apikey <apikey>
#
#   Happyrobot Inc. (c) 2023. All rights reserved.
############################################################################################################################

# Install cv2
# pip install opencv-python 


from typing import Dict, Any
import time
import json
import fire
import requests
from pathlib import Path

from PIL import Image
import cv2
import numpy as np
from pycocotools import mask as coco_mask_utils


def assert_is_uncompressed_rle(rle: Dict[str, Any]):
    assert sorted(["counts", "size"]) == sorted(list(rle.keys())), \
        "Annotation in RLE format does not have counts or size"


def rle_to_mask(rle: dict, order: str = "F") -> np.ndarray:
    """Converts a COCOs run-length encoding (RLE) to binary mask.
    :param rle: Mask in RLE format, that can either be Dict['counts': list[int], 'size] or string
    :return: a 2D binary numpy array where '1's represent the object
    """
    assert_is_uncompressed_rle(rle)
    counts: list = rle.get("counts")
    mask = None
    if isinstance(counts, str):
        mask = coco_mask_utils.decode(rle).astype(bool)
    else:
        binary_array = np.zeros(np.prod(rle.get("size")), dtype=bool)
        start = 0
        for i in range(len(counts) - 1):
            start += counts[i]
            end = start + counts[i + 1]
            binary_array[start:end] = (i + 1) % 2
        mask = binary_array.reshape(*rle.get("size"), order=order)

    # Make sure we don't have an empty mask
    assert np.sum(mask) >= 1, f"Mask is empty for rle: {rle}"
    return mask


def run_inference_on_image(
    image_path: str,
    endpoint: str,
    apikey: str
):
    # Assert image exists
    if not Path(image_path).exists():
        raise ValueError(f"Image {image_path} does not exist")
    
    # Calculate time to run inference
    start_time = time.time()
    print()
    print("#" * 80)
    print(f"Running inference on {Path(image_path).name}")

    # Read image and run inference
    image = open(image_path, 'rb').read()
    res = requests.post(
        endpoint, files={"data": image}, headers={"apikey": apikey}
    )

    # Manage errors
    if res.status_code != 200:
        raise ValueError(f"Error: {res.status_code} - {res.text}")
    elif res.status_code == 401:
        raise ValueError("Error: unauthorized")
    if len(res.content) == 0:
        raise ValueError("Error: empty response")

    # Print time to run inference
    print(f"Time to run inference: {time.time() - start_time:.2f}s")

    # Parse response and return
    data: dict = json.loads(res.content)[0]
    return data


def print_result_summary(data: dict):
    # Get predictions
    predictions = data["predictions"]
    # Get names
    names = data["names"]
    # Print predictions and save dictionary
    num_pred_by_name = {}
    for pred in predictions:
        cls_name = names[pred["cls"]]
        num_pred_by_name[cls_name] = num_pred_by_name.get(cls_name, 0) + 1

    for cls_name, num_pred in num_pred_by_name.items():
        print(f"Found {num_pred} {cls_name}")

    if len(num_pred_by_name) == 0:
        print("No predictions found - make sure you are using the correct endpoint and API key)")
        
    print("#" * 80)


def save_predictions_to_coco_json(
    data: dict,
    json_path: str,
):
    coco_annotations = []
    for pred in data["predictions"]:
        bbox = np.around(pred["bbox"]).astype(int).tolist()
        rle = pred["rle"]
        cls = pred["cls"]
        # Round score to the 3rd decimal
        score = np.around(pred["score"], 3)

        coco_annotations.append({
            "bbox": bbox,
            "segmentation": rle,
            "category_id": cls,
            "category_name": data["names"][pred["cls"]],
            "score": score,
        })

    # Save json
    with open(json_path, "w") as f:
        json.dump(coco_annotations, f, indent=4)

    print(f"Saved json to {json_path}")
    

def save_or_visualize(
    data: dict,
    image_path: str,
    save_images: bool = False,
    visualize: bool = False,
    images_folder: str = None,
    alpha: float = 0.5,
):
    img = cv2.imread(str(image_path))
    img_shape = img.shape[:2]
    colors = np.random.randint(0, 255, size=(len(data["predictions"]), 3)).tolist() 

    # If image shape is half the predicted shape, resize
    predicted_shape = data["predictions"][0]["rle"]["size"]
    if img_shape[0] == predicted_shape[0] // 2 and img_shape[1] == predicted_shape[1] // 2:
        img = cv2.resize(img, (predicted_shape[1], predicted_shape[0]))
    else:
        print(f"Image shape {img_shape} is not half the predicted shape {predicted_shape}")
        print("Report this issue to the Happyrobot team in your support channel")
        print("Cannot visualize predictions")
        return

    bboxes = []
    for color, pred in zip(colors, data["predictions"]):
        bbox = np.around(pred["bbox"]).astype(int)
        bboxes.append(bbox)

        # Draw bounding box
        cv2.rectangle(img, bbox[:2], np.add(bbox[2:], bbox[:2]), color=color, thickness=1)

        # Prepare mask
        mask = rle_to_mask(pred["rle"])

        # Show mask
        img[mask] = np.array(color) * alpha + img[mask] * (1 - alpha)

    for color, pred, bbox in zip(colors, data["predictions"], bboxes):
        cls_name = data["names"][pred["cls"]]
        cv2.putText(
            img, cls_name, (bbox[0], bbox[1]-3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=color, thickness=1
        )

    if save_images:
        cv2.imwrite(str(images_folder / Path(image_path).name), img)
        print(f"Saved image to {images_folder / Path(image_path).name}")

    if visualize:
        # Show image with pillow
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show()
        # Wait for key press
        input("Press Enter to continue...")
        

def main(
    image_path_or_folder: str,
    output_folder: str = "outputs/api_inference",
    endpoint: str = None,
    apikey: str = None,
    save_coco_json: bool = False,
    skip_existing: bool = False,
    save_images: bool = False,
    visualize: bool = False,
    verbose: bool = False,
):
    # Assert endpoint is provided
    assert endpoint is not None, "endpoint must be provided"
    # Assert apikey is provided
    assert apikey is not None, "apikey must be provided"

    print()
    print("Running inference against API")
    print("Endpoint:", endpoint)
    print("API Key:", apikey)
    input("Press Enter to continue...")

    output_folder = Path(output_folder) / Path(image_path_or_folder).parent.name
    images_folder = output_folder / "images"
    annotations_folder = output_folder / "annotations"

    # Update output folder to be a subfolder of the image folder
    if save_images or save_coco_json:
        # Create output folder if it doesn't exist
        output_folder.mkdir(parents=True, exist_ok=True)

        if save_images:
            # Create images folder
            images_folder.mkdir(parents=True, exist_ok=True)
        
        if save_coco_json:
            # Create annotations folder
            annotations_folder.mkdir(parents=True, exist_ok=True)

    # If image_path_or_folder is a folder, run inference on all images in the folder
    if Path(image_path_or_folder).is_dir():
        # Go over all images in the folder (accept any image format supported by OpenCV)
        # Use glob to get all images in the folder of types jpg, png, tiff, etc.
        image_paths = list(Path(image_path_or_folder).glob("*"))
        # Filter out non-image files
        image_paths = [str(image_path) for image_path in image_paths if str(image_path).lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]
        for image_path in image_paths:
            # Skip existing
            json_path = annotations_folder / Path(image_path).with_suffix(".json").name
            if skip_existing and json_path.is_file():
                print(f"Skipping {image_path} because {json_path} already exists")
                continue
            # Run inference on image
            data = run_inference_on_image(str(image_path), endpoint, apikey)
            # Print results
            if verbose:
                print_result_summary(data)
            # Save json
            if save_coco_json:
                save_predictions_to_coco_json(data, json_path)
            # Save or visualize
            if save_images or visualize:
                save_or_visualize(data, image_path, save_images, visualize, images_folder)
    elif Path(image_path_or_folder).is_file():
        # Run inference on image
        data = run_inference_on_image(image_path_or_folder, endpoint, apikey)
        # Print results
        if verbose:
            print_result_summary(data)
        # Save json
        if save_coco_json:
            json_path = annotations_folder / Path(image_path_or_folder).with_suffix(".json").name
            save_predictions_to_coco_json(data, json_path)
        # Save or visualize
        if save_images or visualize:
            save_or_visualize(data, image_path_or_folder, save_images, visualize, images_folder)
    else:
        raise ValueError(
            f"image_path_or_folder {image_path_or_folder} is not a file or folder")


if __name__ == "__main__":
    fire.Fire(main)
