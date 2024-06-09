import logging
import pathlib
import sys
from argparse import ArgumentParser

import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def main(*args):
    parser = ArgumentParser(
        prog="Convert-pt-to-NCNN",
    )
    parser.add_argument('filename')
    parse_args = sys.argv
    if args:
        parse_args = args
    values = parser.parse_args()
    filename = pathlib.Path(values.filename)
    if not filename.exists():
        logger.error(f"File \"{filename.absolute()}\" does not exist")
        return
    try:
        model = YOLO(str(filename.absolute()))
        # model.export(format='ncnn', imgsz=320)
        # model.export(format='onnx', imgsz=320)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    pass


if __name__ == '__main__':
    main()
