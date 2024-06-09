import logging
import os
import pathlib
import shutil
import zipfile

import requests
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics


def download_data_API_KEY():
    from roboflow import Roboflow
    API_KEY = os.environ.get('ROBOFLOW_API_KEY', None)
    if API_KEY is None:
        API_KEY = input("ROBOFLOW API_KEY: ")
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace("testworkspace-eu5g3").project("flat")
    version = project.version(3)
    version.export()
    dataset = version.download("yolov8")
    return dataset


def download_data_zip():
    url = input("DATASET_URL_DOWNLOAD: ")
    response = requests.get(url)
    with open('roboflow-dataset.zip', 'wb') as f:
        f.write(response.content)
    unzip_folder = pathlib.Path('./datasets/roboflow-dataset')
    with zipfile.ZipFile('roboflow-dataset.zip', 'r') as zip_ref:
        zip_ref.extractall(unzip_folder)
    os.unlink('roboflow-dataset.zip')
    return unzip_folder


def is_yes(choice: str) -> bool:
    if choice.lower() == 'y':
        return True
    if choice == '1':
        return True
    return False


def test_speed_on_image():
    # model: nn.Module = torch.load("./trained/best.pth")
    # torch.save(model.state_dict(), "./trained/best.pt")
    # random image from datasets/ folder
    img_list = list(pathlib.Path("./datasets/roboflow-dataset").rglob("*.jpg"))
    img_path = str(img_list[0].absolute())
    model = YOLO("./trained/best.pt", verbose=False)  # load a trained
    for i in tqdm(range(1000)):
        results = model(img_path,
                        verbose=False)


def main():
    dataset_path = pathlib.Path("./datasets/roboflow-dataset/data.yaml")
    if not dataset_path.exists():
        dataset_folder = download_data_zip()
        if dataset_folder is not None:
            dataset_path = pathlib.Path(dataset_folder) / 'data.yaml'
    else:
        logging.getLogger().info(f"Using existing dataset {dataset_path}")
    # Load a trained
    model = YOLO("yolov8n.pt")  # load a pretrained trained (recommended for training)

    # Use the trained
    # trainer must resize and grayscale
    if is_yes(os.environ.get('TUNE', "0")):
        model.tune(data=str(dataset_path.absolute()), epochs=30, iterations=10, optimizer="AdamW", plots=True, save=False,
                   val=False)
        return
    train_result: DetMetrics | dict = model.train(data=str(dataset_path.absolute()),
                                                  epochs=100, save_period=10, save=True,
                                                  imgsz=320, dfl=1.7)  # train the trained
    train_folder = train_result.save_dir
    results = model.val()  # evaluate trained performance on the validation set

    results = model(["https://ultralytics.com/images/bus.jpg"])  # predict on an image
    file_location = pathlib.Path("./trained/best.pt")
    choice = input(f"Save best model to {file_location.absolute()}? [y/n]: ")
    if is_yes(choice):
        if not file_location.parent.exists():
            file_location.parent.mkdir(parents=True)
        shutil.copy(train_folder / 'weights' / "best.pt", file_location)
        # model.export(format="torchscript")
        # success = model.export(format="onnx")  # export the trained to ONNX format
    test_speed_on_image()


if __name__ == '__main__':
    main()
