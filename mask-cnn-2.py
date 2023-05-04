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
    # Loop over masks and compute center point and principal axes for each one
    for i in range(output['masks'].shape[0]):
        mask_coords = output['masks'][i].squeeze(0).cpu().detach().numpy().nonzero()
        mask_x = np.mean(mask_coords[1])
        mask_y = np.mean(mask_coords[0])
        center = (int(mask_x), int(mask_y))
        endpoint1 = (center[0] + 20, center[1])
        endpoint2 = (center[0], center[1] + 20)
        # Draw mask and principal axes on image
        color = (0, 255, 0) # green
        thickness = 2
        cv2.drawContours(image, [np.transpose(np.vstack((mask_coords[1], mask_coords[0]))).reshape((-1, 1, 2)).astype(np.int32)], -1, color, thickness)
        cv2.line(image, center, endpoint1, (255, 0, 0), 2)
        cv2.line(image, center, endpoint2, (0, 0, 255), 2)

# Define function to detect objects in a frame and draw bounding boxes and labels
def detect_objects(frame):
    # Convert frame to PIL Image
    image = PIL.Image.fromarray(frame)
    # Preprocess image and pass through Mask R-CNN model
    image_tensor = torchvision.transforms.functional.to_tensor(image).to(device)
    output = model([image_tensor])[0]
    # Extract bounding boxes and labels from output
    boxes = output['boxes'].cpu().detach().numpy()
    labels = output['labels'].cpu().detach().numpy()
    scores = output['scores'].cpu().detach().numpy()
    # Draw bounding boxes andlabels on frame and compute center point and principal axes of masks
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            x1, y1, x2, y2 = box.astype(int)
            # Check if mask exists for this label
            if label < output['masks'].shape[0]:
                class_name = coco_names[label]
                display_text = f'{class_name}: {score:.2f}'
                color = (255, 255, 255) # white
                thickness = 2
                font_scale = 0.5
                # Draw bounding box, background, and text
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                text_size, _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(frame, (x1, y1 - text_size[1]), (x1 + text_size[0], y1), color, -1)
                cv2.putText(frame, display_text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                # Compute center point and principal axes of mask
                mask_coords = output['masks'][label].squeeze(0).cpu().detach().numpy().nonzero()
                mask_x = np.mean(mask_coords[1])
                mask_y = np.mean(mask_coords[0])
                mask_center = (int(mask_x), int(mask_y))
                # Draw mask and principal axes
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

    # Resize frame to reduce computation time
    frame = cv2.resize(frame, (640, 480))

    # Detect objects in frame and display result
    result_frame = detect_objects(frame)
    cv2.imshow('Mask R-CNN', result_frame)

    # Exit loop if 'q' key is pressed
    key = cv2.waitKey(1) # Wait for 500 ms (0.5 sec)
    if key == ord('q'):
        break
# Release resources
cap.release()
cv2.destroyAllWindows()