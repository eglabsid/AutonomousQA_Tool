import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def segment_and_get_contours_with_mrcnn(image_path, threshold=0.5):
    # 1. Load pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # 2. Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)

    # 3. Get predictions
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract masks and scores
    masks = predictions[0]['masks']
    scores = predictions[0]['scores']

    # Select masks above threshold
    selected_masks = masks[scores > threshold]
    contours_list = []

    # Process each mask
    for mask in selected_masks:
        binary_mask = (mask[0].cpu().numpy() > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_list.append(contours)

    return image, contours_list, selected_masks

def visualize_with_contours(image, contours_list):
    # Convert PIL image to OpenCV format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw contours on the image
    for contours in contours_list:
        for contour in contours:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    # Display the image with contours
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Segmented Image with Contours")
    plt.show()

if __name__ == "__main__":
    # Path to image
    image_path = r"./video/cp.jpg"

    # Get segmentation and contours
    image, contours_list, masks = segment_and_get_contours_with_mrcnn(image_path)

    # Visualize results
    visualize_with_contours(image, contours_list)

    # Print contour coordinates
    for i, contours in enumerate(contours_list):
        print(f"Mask {i}:")
        for contour in contours:
            print(contour.reshape(-1, 2))  # Output contour coordinates