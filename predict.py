import pathlib

from ultralytics import YOLO

img_list = list(pathlib.Path("./datasets/roboflow-dataset").rglob("*.jpg"))
img_path = str(img_list[0].absolute())
# Load a model
model = YOLO("./trained/best.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model([img_path])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk