import cv2
import numpy as np
import PIL
import torch
import torchvision

# Load the Mask R-CNN model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# Define the classes for COCO dataset used in Mask R-CNN
coco_names = [
    'unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack',
    'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
    'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Define function to draw masks on image
def draw_masks(image, output):
    masks = None
    for score, mask in zip(output['scores'], output['masks']):
        if score > 0.5:
            if masks is None:
                masks = mask
            else:
                masks = torch.max(masks, mask)

    if masks is not None:
        masks = masks.squeeze(0).cpu().detach().numpy()
        masks = (masks > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):
            color = (0, 255, 0) # green
            cv2.drawContours(image, contour, -1, color, 3)
# Define function to detect objects in a frame and draw bounding boxes and labels
def detect_objects(frame):
    # Convert frame to PIL Image
    image = PIL.Image.fromarray(frame)
    # Preprocess image and pass through Mask R-CNN model
    image_tensor = torchvision.transforms.functional.to_tensor(image).to(device)
    output = model([image_tensor])[0]
    # Extract bounding boxes and labels from output
    boxes = output['boxes'].detach().numpy()
    labels = output['labels'].detach().numpy()
    scores = output['scores'].detach().numpy()
    # Draw bounding boxes and labels on frame
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            x1, y1, x2, y2 = box.astype(int)
            class_name = coco_names[label]
            display_text = f'{class_name}: {score:.2f}'
            color = (255, 255, 255) # white
            thickness = 2
            font_scale = 0.5
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            # Draw label background
            text_size, _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1 - text_size[1]), (x1 + text_size[0], y1), color, -1)
            # Draw label text
            cv2.putText(frame, display_text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    # Draw masks on image
    draw_masks(frame, output)
    return frame
# Initialize OpenCV window for displaying frames
cv2.namedWindow('Mask R-CNN', cv2.WINDOW_NORMAL)

# Initialize video capture object for webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects in frame and display result
    result_frame = detect_objects(frame)
    cv2.imshow('Mask R-CNN', result_frame)
    
    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()



