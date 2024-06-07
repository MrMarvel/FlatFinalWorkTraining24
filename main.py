import pathlib

import torch
from torch import nn
from tqdm import tqdm
from ultralytics import YOLO


def is_yes(choice: str) -> bool:
    if choice.lower() == 'y':
        return True
    if choice == '1':
        return True
    return False
def main2():
    # model: nn.Module = torch.load("./trained/best.pth")
    # torch.save(model.state_dict(), "./trained/best.pt")
    model = YOLO("./trained/best.pt", verbose=False)  # load a trained
    for i in tqdm(range(10000), smoothing=0.01):
        results = model("./datasets/Flat-2/valid/1b160caee5ca_jpg.rf.3ce8014f8311e91f179c43ce1a12f101.jpg",
                        verbose=False)
def main():
    # Load a trained
    model = YOLO("yolov8n.pt")  # load a pretrained trained (recommended for training)

    # Use the trained
    dataset_path = pathlib.Path("./datasets/Flat-3/data.yaml")
    # trainer must resize and grayscale
    results = model.train(data=str(dataset_path.absolute()), epochs=3)  # train the trained
    results = model.val()  # evaluate trained performance on the validation set

    results = model(["https://ultralytics.com/images/bus.jpg"])  # predict on an image
    file_location = pathlib.Path("./trained/best.pt")
    choice = input(f"Save best model to {file_location.absolute()}? [y/n]: ")
    if is_yes(choice):
        model.save(file_location)
        model.export(format="torchscript")
        success = model.export(format="onnx")  # export the trained to ONNX format


if __name__ == '__main__':
    main2()

